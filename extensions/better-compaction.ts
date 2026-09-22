/**
 * better-compaction.ts — Better Compaction: compaction that appends the prompt
 * instead of pre-filling from a system prompt.
 *
 * The built-in compaction builds a one-off summarization request that looks
 * nothing like the conversation it summarizes:
 *
 *   [summarization system prompt] + [whole conversation serialized as ONE user message]
 *
 * That request shares no prefix with the last normal agent request, so the
 * backend has to prefill the entire conversation from scratch every time.
 *
 * This extension replaces it with:
 *
 *   [the agent's own request, truncated at the compaction cut point — system
 *    prompt + conversation up to the first kept entry, as real messages] +
 *    [ONE user message at the very end containing the compaction prompt]
 *
 * Because that prefix is byte-identical to a prefix of the previous agent
 * request, any backend with prefix/KV caching (llama.cpp llama-server,
 * Anthropic, OpenAI, ...) reuses the cached prefix and only prefills the
 * appended message (plus anything that arrived after the last request). The
 * summary is generated much faster, and the model sees the real conversation
 * instead of a serialized transcript.
 *
 * The kept tail (the most recent ~keepRecentTokens of messages that pi retains
 * verbatim after compaction) is deliberately NOT sent: the summarizer would
 * just re-describe messages that remain in the session, bloating the summary.
 *
 * The summarization call also mirrors the settings pi applies to normal agent
 * requests (images.blockImages, retry policy, provider timeout/transport,
 * thinking budgets), reading the same settings files pi merges, so the
 * request stays identical to what the agent sends.
 *
 * Configuration (pi's settings.json, global or project; project wins):
 *
 *   "betterCompaction": {
 *     "enabled": true,   // false → extension stays silent, pi's built-in
 *                        // compaction runs instead. Settings are read at
 *                        // compaction time, so a change applies at the next
 *                        // compaction without restarting.
 *     "report": true,    // false → don't show the "Compaction completed in
 *                        // Xs ..." message after a compaction
 *     "thinking": "low"  // "low" (default) | "off" | "inherit" (the session's
 *                        // level — the built-in behavior). Defaults to "low":
 *                        // summaries are short structured output, and a high
 *                        // thinking level can hit the model's thinking cap
 *                        // mid-summary and stop the compaction. "off" omits
 *                        // the reasoning parameter entirely, same as the
 *                        // agent's own requests when session thinking is off.
 *   }
 *
 * Both options can also be managed at runtime with the /bc command:
 *
 *   /bc                       show effective options and which settings file
 *                             provides them
 *   /bc on | /bc off          enable / disable the extension (the built-in
 *                             compaction takes over when disabled)
 *   /bc thinking low|off|inherit
 *                             set the summarization thinking level
 *   /bc report on|off       show / hide the post-compaction report message
 *                             (timing and cache read)
 *
 * /bc writes to the settings file that currently provides the
 * betterCompaction section (project file if it defines one, otherwise the
 * global file), so it always edits the file that is in effect.
 *
 * When compaction runs without thinking, a warning is shown once per session
 * (some models summarize significantly worse — or not at all — without
 * thinking). After each successful compaction, a message reports total time,
 * the thinking vs generation split, and how many tokens were served from the
 * prefix cache.
 *
 * Everything else works exactly like normal compaction: same summary format,
 * same firstKeptEntryId / tokensBefore bookkeeping, same cumulative file
 * tracking, same session_compact / session_compact_failed events, same
 * kept-messages rebuild. The extension only supplies the summary via the
 * `session_before_compact` event; pi does the rest with its normal path.
 *
 * Note: the summarization model now sees the conversation as a live dialogue,
 * so the appended prompt explicitly forbids continuing the conversation.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	buildContextEntries,
	CONFIG_DIR_NAME,
	convertToLlm,
	getAgentDir,
	sessionEntryToContextMessages,
} from "@earendil-works/pi-coding-agent";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import { contentText, normalizeContext, retryAssistantCall } from "@earendil-works/pi-ai";

// ---------------------------------------------------------------------------
// Prompts (verbatim from pi's built-in compaction, compaction.ts / utils.ts)
// ---------------------------------------------------------------------------

/**
 * The role instruction that the built-in compaction sends as its system
 * prompt. Here it becomes part of the appended user message at the end of
 * the conversation.
 */
const SUMMARY_ROLE = `You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.`;

const SUMMARIZATION_PROMPT = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

const UPDATE_SUMMARIZATION_PROMPT = `The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

// ---------------------------------------------------------------------------
// Helpers (same semantics as pi's compaction/utils.ts)
// ---------------------------------------------------------------------------

/** Same as pi's computeFileLists(): readFiles = read but not modified. */
function computeFileLists(fileOps: { read: Set<string>; written: Set<string>; edited: Set<string> }): {
	readFiles: string[];
	modifiedFiles: string[];
} {
	const modified = new Set([...fileOps.edited, ...fileOps.written]);
	const readFiles = [...fileOps.read].filter((f) => !modified.has(f)).sort();
	const modifiedFiles = [...modified].sort();
	return { readFiles, modifiedFiles };
}

/** Same as pi's formatFileOperations(): XML sections appended to the summary. */
function formatFileOperations(readFiles: string[], modifiedFiles: string[]): string {
	const sections: string[] = [];
	if (readFiles.length > 0) {
		sections.push(`<read-files>\n${readFiles.join("\n")}\n</read-files>`);
	}
	if (modifiedFiles.length > 0) {
		sections.push(`<modified-files>\n${modifiedFiles.join("\n")}\n</modified-files>`);
	}
	if (sections.length === 0) return "";
	return `\n\n${sections.join("\n\n")}`;
}

// ---------------------------------------------------------------------------
// pi settings
//
// pi's agent applies several settings to every LLM request (image blocking,
// retry policy, provider timeout/transport, thinking budgets). The
// compaction request must mirror them to stay identical to the agent's own
// requests. Extensions have no settings API, so we read the same files pi's
// SettingsManager merges (global, then project; project wins).
// ---------------------------------------------------------------------------

interface PiSettings {
	images?: { blockImages?: boolean };
	retry?: {
		enabled?: boolean;
		maxRetries?: number;
		baseDelayMs?: number;
		maxAgentDelayMs?: number;
		provider?: {
			timeoutMs?: number;
			maxRetries?: number;
			maxRetryDelayMs?: number;
		};
	};
	httpIdleTimeoutMs?: number;
	websocketConnectTimeoutMs?: number;
	transport?: "sse" | "websocket" | "websocket-cached" | "auto";
	thinkingBudgets?: { minimal?: number; low?: number; medium?: number; high?: number };
	betterCompaction?: {
		enabled?: boolean;
		/** Show the "Compaction completed in Xs ..." message after compaction. */
		report?: boolean;
		/** Thinking level for the summarization call. */
		thinking?: "low" | "off" | "inherit";
	};
}

function isObject(value: unknown): value is Record<string, unknown> {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Same deep-merge semantics as pi's SettingsManager (override wins). */
function mergeSettings(base: Record<string, unknown>, override: Record<string, unknown>): Record<string, unknown> {
	const out: Record<string, unknown> = { ...base };
	for (const [key, value] of Object.entries(override)) {
		out[key] = isObject(out[key]) && isObject(value) ? mergeSettings(out[key], value) : value;
	}
	return out;
}

function readSettingsFile(path: string): Record<string, unknown> {
	try {
		const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
		return isObject(parsed) ? parsed : {};
	} catch {
		return {};
	}
}

function globalSettingsPath(): string {
	return join(getAgentDir(), "settings.json");
}

function projectSettingsPath(cwd: string): string {
	return join(cwd, CONFIG_DIR_NAME, "settings.json");
}

interface PiSettingsSource {
	settings: PiSettings;
	/** Which file currently provides the effective betterCompaction config. */
	betterCompactionSource: "global" | "project";
	betterCompactionSourcePath: string;
}

/** Read pi's global + project settings the same way SettingsManager does. */
function loadPiSettings(cwd: string): PiSettingsSource {
	const globalSettings = readSettingsFile(globalSettingsPath());
	const projectSettings = readSettingsFile(projectSettingsPath(cwd));
	const fromProject = "betterCompaction" in projectSettings;
	return {
		settings: mergeSettings(globalSettings, projectSettings) as PiSettings,
		betterCompactionSource: fromProject ? "project" : "global",
		betterCompactionSourcePath: fromProject ? projectSettingsPath(cwd) : globalSettingsPath(),
	};
}

/**
 * Update the betterCompaction section in the settings file that currently
 * provides it (project wins over global), so /bc always edits the file that
 * is actually in effect. Returns the path written.
 */
function writeBetterCompactionSetting(
	cwd: string,
	patch: { enabled?: boolean; report?: boolean; thinking?: "low" | "off" | "inherit" },
): string {
	const { betterCompactionSourcePath } = loadPiSettings(cwd);
	const current = readSettingsFile(betterCompactionSourcePath);
	const section = isObject(current.betterCompaction) ? { ...current.betterCompaction } : {};
	Object.assign(section, patch);
	current.betterCompaction = section;
	writeFileSync(betterCompactionSourcePath, JSON.stringify(current, null, 2) + "\n");
	return betterCompactionSourcePath;
}

type LlmMessage = ReturnType<typeof convertToLlm>[number];

const IMAGE_BLOCKED_PLACEHOLDER = "Image reading is disabled.";

/**
 * Same as pi's convertToLlmWithBlockImages filter (sdk.ts): when
 * images.blockImages is enabled, the agent's requests replace image blocks
 * with a text placeholder. Apply the identical filter so the rebuilt prefix
 * matches what the agent actually sent to the LLM.
 */
function applyBlockImages(messages: LlmMessage[]): LlmMessage[] {
	return messages.map((msg) => {
		if (msg.role !== "user" && msg.role !== "toolResult") return msg;
		const content = msg.content;
		if (typeof content === "string" || !content.some((c) => c.type === "image")) return msg;
		const filtered = content
			.map((c) => (c.type === "image" ? { type: "text" as const, text: IMAGE_BLOCKED_PLACEHOLDER } : c))
			.filter(
				(c, i, arr) =>
					!(
						c.type === "text" &&
						c.text === IMAGE_BLOCKED_PLACEHOLDER &&
						i > 0 &&
						arr[i - 1].type === "text" &&
						(arr[i - 1] as { text?: string }).text === IMAGE_BLOCKED_PLACEHOLDER
					),
			);
		return { ...msg, content: filtered } as LlmMessage;
	});
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI) {
	// Warn once per session when compaction will run without thinking.
	pi.on("session_start", (_event, ctx) => {
		const { settings } = loadPiSettings(ctx.cwd);
		if (settings.betterCompaction?.enabled === false) return;
		const thinking = settings.betterCompaction?.thinking ?? "low";
		const noThinking =
			thinking === "off" || (thinking === "inherit" && (!ctx.thinkingLevel || ctx.thinkingLevel === "off"));
		if (noThinking && ctx.hasUI) {
			ctx.ui.notify(
				"Better Compaction: thinking is disabled for compaction — some models summarize significantly worse (or not at all) without it",
				"warning",
			);
		}
	});

	// /bc — inspect and control this extension's options.
	pi.registerCommand("bc", {
		description: "Better Compaction: /bc [on|off|thinking low|off|inherit|report on|off] — show or change compaction options",
		getArgumentCompletions: (prefix) => {
			const options = ["on", "off", "thinking", "report"];
			const filtered = options.filter((o) => o.startsWith(prefix.toLowerCase()));
			return filtered.length > 0
				? filtered.map((o) => ({ value: o, label: o, description: "Better Compaction option" }))
				: null;
		},
		handler: async (args, ctx) => {
			const { settings, betterCompactionSource, betterCompactionSourcePath } = loadPiSettings(ctx.cwd);
			const section = settings.betterCompaction ?? {};
			const enabled = section.enabled !== false;
			const report = section.report !== false;
			const thinking = section.thinking ?? "low";
			const arg = args.trim().toLowerCase();

			if (arg === "") {
				ctx.ui.notify(
					`Better Compaction: ${enabled ? "enabled" : "disabled"} | compaction thinking: ${thinking} | ` +
						`report: ${report ? "on" : "off"} | session thinking: ${ctx.thinkingLevel ?? "off"} | ` +
						`config: ${betterCompactionSource} (${betterCompactionSourcePath})`,
				);
				return;
			}
			if (arg === "on" || arg === "off") {
				const path = writeBetterCompactionSetting(ctx.cwd, { enabled: arg === "on" });
				ctx.ui.notify(
					`Better Compaction ${arg === "on" ? "enabled" : "disabled"} (written to ${path}; ` +
						`applies at the next compaction)`,
				);
				return;
			}
			if (arg.startsWith("thinking")) {
				const value = arg.split(/\s+/)[1];
				if (value !== "low" && value !== "off" && value !== "inherit") {
					ctx.ui.notify("Better Compaction: usage: /bc thinking low|off|inherit", "error");
					return;
				}
				const path = writeBetterCompactionSetting(ctx.cwd, { thinking: value });
				ctx.ui.notify(
					`Better Compaction: compaction thinking set to ${value} (written to ${path}; ` +
						`applies at the next compaction)`,
				);
				return;
			}
			if (arg.startsWith("report")) {
				const value = arg.split(/\s+/)[1];
				if (value !== "on" && value !== "off") {
					ctx.ui.notify("Better Compaction: usage: /bc report on|off", "error");
					return;
				}
				const path = writeBetterCompactionSetting(ctx.cwd, { report: value === "on" });
				ctx.ui.notify(
					`Better Compaction: post-compaction report ${value === "on" ? "enabled" : "disabled"} ` +
						`(written to ${path}; applies at the next compaction)`,
				);
				return;
			}
			ctx.ui.notify("Better Compaction: usage: /bc [on|off|thinking low|off|inherit|report on|off]", "error");
		},
	});

	pi.on("session_before_compact", async (event, ctx) => {
		const { preparation, customInstructions, signal } = event;
		const model = ctx.model;
		if (!model) return; // no model selected — let pi's normal path report it

		// The whole extension can be switched off via settings; pi's built-in
		// compaction takes over. Read at compaction time so no restart is needed.
		const { settings } = loadPiSettings(ctx.cwd);
		if (settings.betterCompaction?.enabled === false) return;

		// 1. Rebuild the prefix of pi's LLM request up to the compaction cut.
		//
		// preparation.firstKeptEntryId marks where pi starts KEEPING messages
		// verbatim; everything before it is what gets summarized and discarded.
		// The agent's last request was [system prompt + full conversation], and
		// the conversation up to the cut is a prefix of that, so prefix/KV
		// caches still cover it. The kept tail is deliberately omitted: the
		// summarizer must not re-describe messages that stay in the session.
		const entries = ctx.sessionManager.getEntries();
		const contextEntries = buildContextEntries(entries, ctx.sessionManager.getLeafId());
		const cutIndex = contextEntries.findIndex((e) => e.id === preparation.firstKeptEntryId);
		if (cutIndex < 1) return; // nothing to summarize — let pi decide
		const prefixMessages = contextEntries.slice(0, cutIndex).flatMap(sessionEntryToContextMessages);
		let llmMessages = convertToLlm(prefixMessages);

		// pi's agent filters images out of every request when images.blockImages
		// is enabled (replacing them with a placeholder). Mirror that so the
		// rebuilt prefix matches what the agent actually sent to the LLM.
		if (settings.images?.blockImages) {
			llmMessages = applyBlockImages(llmMessages);
		}

		// A dangling assistant tool call (aborted mid-turn) is not a valid
		// request on most providers; strip trailing tool calls so the request
		// stays well-formed. Only the last message is affected.
		while (llmMessages.length > 0) {
			const last = llmMessages[llmMessages.length - 1];
			if (last.role === "assistant" && last.content.some((b) => b.type === "toolCall")) {
				const kept = last.content.filter((b) => b.type !== "toolCall");
				if (kept.length === 0) llmMessages.pop();
				else llmMessages[llmMessages.length - 1] = { ...last, content: kept };
				continue;
			}
			break;
		}
		if (llmMessages.length === 0) return; // nothing to compact — let pi decide

		// 2. The compaction prompt: pi's summarization system prompt, now
		//    appended at the end of the conversation as a single user message.
		let promptText = SUMMARY_ROLE + "\n\n";
		if (preparation.previousSummary) {
			promptText += `<previous-summary>\n${preparation.previousSummary}\n</previous-summary>\n\n`;
		}
		promptText += preparation.previousSummary ? UPDATE_SUMMARIZATION_PROMPT : SUMMARIZATION_PROMPT;
		if (customInstructions) {
			promptText += `\n\nAdditional focus: ${customInstructions}`;
		}

		const requestContext = normalizeContext({
			messages: [
				...llmMessages,
				{
					role: "user",
					content: [{ type: "text", text: promptText }],
					timestamp: Date.now(),
				},
			],
		});

		// 3. One LLM call through the configured provider. Same token budget
		//    and thinking level as the built-in compaction, and the same
		//    session id the agent uses so cache-aware backends route to the
		//    same cache. Cache retention stays at the default ("short") — we
		//    want the read, unlike the built-in one-off which uses "none".
		const maxTokens = Math.min(
			Math.floor(0.8 * preparation.settings.reserveTokens),
			model.maxTokens > 0 ? model.maxTokens : Number.POSITIVE_INFINITY,
		);
		// Mirror pi's request wrapper (sdk.ts buildRequestOptions): the agent's
		// requests carry the provider timeout/transport/retry settings from pi's
		// settings, so the compaction request uses the same values.
		const providerRetry = settings.retry?.provider;
		const httpIdleTimeoutMs =
			typeof settings.httpIdleTimeoutMs === "number" ? settings.httpIdleTimeoutMs : 300_000;
		const streamOptions: {
			maxTokens: number;
			signal: AbortSignal;
			sessionId: string;
			cacheRetention: "short";
			reasoning?: Exclude<typeof ctx.thinkingLevel, "off">;
			timeoutMs: number;
			maxRetryDelayMs: number;
			maxRetries?: number;
			websocketConnectTimeoutMs?: number;
			transport?: PiSettings["transport"];
			thinkingBudgets?: PiSettings["thinkingBudgets"];
		} = {
			maxTokens,
			signal,
			sessionId: ctx.sessionManager.getSessionId(),
			cacheRetention: "short",
			timeoutMs: providerRetry?.timeoutMs ?? (httpIdleTimeoutMs === 0 ? 2_147_483_647 : httpIdleTimeoutMs),
			maxRetryDelayMs: providerRetry?.maxRetryDelayMs ?? 60_000,
		};
		if (providerRetry?.maxRetries !== undefined) {
			streamOptions.maxRetries = providerRetry.maxRetries;
		}
		if (settings.websocketConnectTimeoutMs !== undefined) {
			streamOptions.websocketConnectTimeoutMs = settings.websocketConnectTimeoutMs;
		}
		if (settings.transport !== undefined) {
			streamOptions.transport = settings.transport;
		}
		if (settings.thinkingBudgets !== undefined) {
			streamOptions.thinkingBudgets = settings.thinkingBudgets;
		}
		// Thinking level for the summarization call. Defaults to "low": the
		// summary is short structured output, and inheriting a high session
		// level is wasteful here and can hit the model's thinking cap
		// mid-summary (stopping the compaction). "off" omits the reasoning
		// parameter — the same request the agent sends when session thinking
		// is off — and "inherit" restores the built-in behavior.
		const thinkingChoice = settings.betterCompaction?.thinking ?? "low";
		if (model.reasoning) {
			if (thinkingChoice === "low") {
				streamOptions.reasoning = "low";
			} else if (thinkingChoice === "inherit" && ctx.thinkingLevel && ctx.thinkingLevel !== "off") {
				streamOptions.reasoning = ctx.thinkingLevel;
			}
		}

		// Same agent-level retry policy pi's built-in compaction uses
		// (settings.retry.*), so the compaction call honors retry.enabled too.
		// The stream is consumed manually (same events .result() uses) so the
		// final attempt can be timed: thinking time is the sum of
		// thinking_start→thinking_end spans, the rest is generation.
		let timing = { totalMs: 0, thinkingMs: 0 };
		const response = await retryAssistantCall(
			async () => {
				const t0 = Date.now();
				let thinkingMs = 0;
				let spanStart: number | null = null;
				let message: AssistantMessage | undefined;
				const stream = await ctx.modelRegistry.streamSimple(model, requestContext, streamOptions);
				for await (const event of stream) {
					if (event.type === "thinking_start" && spanStart === null) {
						spanStart = Date.now();
					} else if (event.type === "thinking_end") {
						if (spanStart !== null) {
							thinkingMs += Date.now() - spanStart;
							spanStart = null;
						}
					} else if (event.type === "done") {
						message = event.message;
					} else if (event.type === "error") {
						message = event.error;
					}
				}
				if (spanStart !== null) thinkingMs += Date.now() - spanStart; // unclosed span
				timing = { totalMs: Date.now() - t0, thinkingMs };
				if (!message) throw new Error("Summarization stream ended without a result");
				return message;
			},
			{
				enabled: settings.retry?.enabled ?? true,
				maxRetries: settings.retry?.maxRetries ?? 3,
				baseDelayMs: settings.retry?.baseDelayMs ?? 2000,
				maxAgentDelayMs: settings.retry?.maxAgentDelayMs ?? 60_000,
			},
			signal,
		);

		// 4. Validate like the built-in compaction does: never persist a
		//    partial or failed summary.
		if (response.stopReason === "aborted") {
			throw new Error("Compaction aborted");
		}
		if (response.stopReason === "error") {
			throw new Error(`Summarization failed: ${response.errorMessage || "Unknown error"}`);
		}
		if (response.stopReason === "length") {
			throw new Error("Summarization failed: generation hit the token cap and the summary is incomplete");
		}
		// The request carries the session's real tool set (needed for the prefix
		// to stay identical to the agent's request), so the summarizer CAN call
		// tools. Never persist a summary that tried to continue the work.
		if (response.content.some((block) => block.type === "toolCall")) {
			throw new Error("Summarization attempted to call a tool");
		}
		const summaryText = contentText(response.content).trim();
		if (!summaryText) {
			throw new Error("Summarization returned no text");
		}

		// 5. Same cumulative file tracking as the built-in compaction.
		const { readFiles, modifiedFiles } = computeFileLists(preparation.fileOps);
		const summary = summaryText + formatFileOperations(readFiles, modifiedFiles);

		// 6. Report timing (measured on the final attempt). The thinking
		//    breakdown is shown only when the model actually thought.
		const details: string[] = [];
		if (timing.thinkingMs > 250) {
			const genMs = Math.max(0, timing.totalMs - timing.thinkingMs);
			details.push(
				`${(timing.thinkingMs / 1000).toFixed(1)}s thinking, ${(genMs / 1000).toFixed(1)}s generating`,
			);
		}
		if (response.usage.cacheRead > 0) {
			const k = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`);
			details.push(`${k(response.usage.cacheRead)} tokens from cache`);
		}
		const suffix = details.length > 0 ? ` (${details.join(", ")})` : "";
		// The post-compaction report can be switched off with report=false.
		if (settings.betterCompaction?.report !== false && ctx.hasUI) {
			ctx.ui.notify(`Compaction completed in ${(timing.totalMs / 1000).toFixed(1)}s${suffix}.`);
		}

		return {
			compaction: {
				summary,
				firstKeptEntryId: preparation.firstKeptEntryId,
				tokensBefore: preparation.tokensBefore,
				usage: response.usage,
				details: { readFiles, modifiedFiles, strategy: "better-compaction" },
			},
		};
	});
}
