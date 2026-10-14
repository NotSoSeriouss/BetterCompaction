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
 *     "report": true,    // false → don't append the "Compaction completed
 *                        // in Xs ..." message to the chat after a compaction
 *     "thinking": "inherit",  // "inherit" (default) | "off". The summarization
 *                             // call always runs at the session's own thinking
 *                             // level — we never change the thinking mode in
 *                             // the request, because with some models the
 *                             // thinking mode is applied at the chat-template
 *                             // level and switching it forces a full prefill
 *                             // of the conversation. Instead, thinking is
 *                             // steered in the appended compaction prompt:
 *                             // "inherit" asks the model to keep thinking
 *                             // brief, "off" asks it to skip thinking entirely
 *                             // and go straight to the summary.
 *   }
 *
 * Both options can also be managed at runtime with the /bc command:
 *
 *   /bc                       show effective options and which settings file
 *                             provides them
 *   /bc on | /bc off          enable / disable the extension (the built-in
 *                             compaction takes over when disabled)
 *   /bc thinking off|inherit  set the summarization thinking level
 *   /bc report on|off         show / hide the post-compaction report message
 *                             (timing and cache read)
 *
 * /bc writes to the settings file that currently provides the
 * betterCompaction section (project file if it defines one, otherwise the
 * global file), so it always edits the file that is in effect.
 *
 * When compaction runs without thinking, a warning is shown once per session
 * (some models summarize significantly worse — or not at all — without
 * thinking). After each successful compaction, a persistent message is
 * appended to the chat reporting total time, the thinking vs generation
 * split, and how many tokens were served from the prefix cache. It is a
 * custom session entry (pi.appendEntry) rendered by an entry renderer: it
 * stays in the transcript — including across session reloads — but never
 * participates in LLM context, so it costs no tokens and no cache. The entry
 * is appended on pi's `session_compact` event, which fires only after pi has
 * applied the compaction result and finished the compaction lifecycle — not
 * inside `session_before_compact`, where it would be appended before the
 * compaction is actually complete (making the message appear inconsistently
 * or disappear depending on the UI lifecycle).
 *
 * Everything else works exactly like normal compaction: same summary format,
 * same firstKeptEntryId / tokensBefore bookkeeping, same cumulative file
 * tracking, same session_compact / session_compact_failed events, same
 * kept-messages rebuild. The extension only supplies the summary via the
 * `session_before_compact` event; pi does the rest with its normal path.
 *
 * Note: the summarization model now sees the conversation as a live dialogue
 * full of real tool calls, so the appended prompt explicitly forbids
 * continuing the conversation and calling tools. Models that attempt a tool
 * call anyway are handled gracefully (see the validation step).
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	buildSessionProjection,
	CONFIG_DIR_NAME,
	convertToLlm,
	getAgentDir,
} from "@earendil-works/pi-coding-agent";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import { contentText, normalizeContext, retryAssistantCall } from "@earendil-works/pi-ai";
import { Text } from "@earendil-works/pi-tui";

// ---------------------------------------------------------------------------
// Prompts (verbatim from pi's built-in compaction, compaction.ts / utils.ts)
// ---------------------------------------------------------------------------

/**
 * The role instruction that the built-in compaction sends as its system
 * prompt. Here it becomes part of the appended user message at the end of
 * the conversation.
 */
const SUMMARY_ROLE = `
<system>
THIS IS NOT A USER MESSAGE: IGNORE WHATEVER YOU WERE DOING YOUR TASK HAS NOW CHANGED. YOU'RE NOT A CODING AGENT ANYMORE, YOU'RE NOW A COMPACTING AGENT.
I KNOW IT MIGHT SEEM THAT YOU WERE DOING SOMETHING ELSE BUT THE MESSAGGES BEFORE ARE JUST FOR CONTEXT, YOU ONLY SUMMARIZE. DON'T USE TOOLS, DON'T DO MORE THAN ONE TURN.
YOU WILL NOW ONLY MAKE THIS SUMMARY.

I WILL MAKE IT AS CLEAR AS POSSIBLE FOR YOU:
DON'T USE ANY TOOL
DON'T WRITE MORE THAN 1 SINGLE MESSAGE WITH THE ENTIRE SUMMARY IN IT
I DON'T CARE THAT WE WERE DOING SOMETHING ELSE
YOUR ONLY TASK IS TO DO THIS SUMMARY RIGHT NOW AND NOTHING ELSE
DO NOT ASK ME ANY OTHER QUESTIONS.
</system>

You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.
`;

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

/**
 * Appended to every compaction prompt (the default "inherit" mode). The
 * summary is short structured output, so we tell the model not to over-invest
 * thinking in it. This replaces the old per-level thinking-token cap: capping
 * the budget would change the request relative to the agent's last request
 * and break the prompt-cache prefix, whereas steering in the prompt does not.
 */
const BRIEF_THINKING_INSTRUCTION = `

Keep your internal thinking short and direct: this is a brief, structured summary of a conversation that has already happened, not a problem to solve. Do not re-examine the conversation in detail or reason at length — think just long enough to write the summary, then stop.`;

/**
 * Appended when the compaction's thinking is "off". We do NOT turn thinking
 * off in the request: with some models the thinking mode is applied at the
 * chat-template level (the template injects different prompts into the running
 * chat), so switching it changes the prompt structure and forces a full
 * prefill of the whole conversation. Instead we keep the request identical to
 * the agent's last request and tell the model, in the prompt, to skip thinking
 * and write the summary straight away.
 */
const NO_THINKING_INSTRUCTION = `

Thinking is disabled for this task. I know it may still look available to you and you may even be thinking right now, but stop thinking immediately: do not deliberate, do not re-read or re-analyze the conversation — go straight to writing the summary.`;

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
	betterCompaction?: BetterCompactionSettings;
}

/** The betterCompaction section of pi's settings. */
interface BetterCompactionSettings {
	enabled?: boolean;
	/** Show the "Compaction completed in Xs ..." message after compaction. */
	report?: boolean;
	/**
	 * Thinking level for the summarization call. "inherit" (default) runs at
	 * the session's own thinking level and steers the model to keep thinking
	 * brief; "off" steers it to skip thinking entirely and go straight to the
	 * summary. In both cases the request keeps the session's thinking mode
	 * (changing it would force a full prefill); thinking is steered in the
	 * appended prompt, not in the request.
	 */
	thinking?: "off" | "inherit";
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
function writeBetterCompactionSetting(cwd: string, patch: Partial<BetterCompactionSettings>): string {
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

/**
 * Timing/cache metadata for the post-compaction report. Stored during the
 * summarization step (session_before_compact) and consumed by the
 * session_compact event, which pi fires only after the compaction result has
 * been applied. Keyed by session id so concurrent sessions don't interfere.
 */
interface CompactionReport {
	/** Wall-clock time of the final summarization attempt, ms. */
	totalMs: number;
	/** First thinking token → start of the answer, ms (see produce() below). */
	thinkingMs: number;
	/** Tokens served from the prefix cache on the final attempt. */
	cacheRead: number;
}

/**
 * Payload persisted with the "better-compaction-report" custom session entry.
 * Custom entries are stored in the session file, rendered in the chat by the
 * entry renderer, and excluded from LLM context — so the report is persistent
 * (it survives session reloads) without costing tokens or cache.
 */
interface CompactionReportData {
	totalMs: number;
	thinkingMs: number;
	cacheRead: number;
	timestamp: number;
}

/** Build the report message (timing measured on the final attempt). */
function formatCompactionReport(report: CompactionReportData): string {
	const details: string[] = [];
	// The thinking breakdown is shown only when the model actually thought.
	if (report.thinkingMs > 250) {
		const genMs = Math.max(0, report.totalMs - report.thinkingMs);
		details.push(`${(report.thinkingMs / 1000).toFixed(1)}s thinking, ${(genMs / 1000).toFixed(1)}s generating`);
	}
	if (report.cacheRead > 0) {
		const k = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`);
		details.push(`${k(report.cacheRead)} tokens from cache`);
	}
	const suffix = details.length > 0 ? ` (${details.join(", ")})` : "";
	return `Compaction completed in ${(report.totalMs / 1000).toFixed(1)}s${suffix}.`;
}

export default function (pi: ExtensionAPI) {
	// Reports stashed by the session_before_compact handler, emitted on
	// session_compact (see below). Cleared on session_compact_failed.
	const pendingReports = new Map<string, CompactionReport>();

	// Render the persisted post-compaction report in the chat. Custom entries
	// never participate in LLM context, so the report stays visible in the
	// transcript without being sent to the model.
	pi.registerEntryRenderer<CompactionReportData>("better-compaction-report", (entry, _options, theme) => {
		const data = entry.data;
		if (!data || typeof data.totalMs !== "number") return undefined;
		return new Text(theme.fg("dim", formatCompactionReport(data)), 0, 0);
	});

	// Warn once per session when compaction will run without thinking.
	pi.on("session_start", (_event, ctx) => {
		const { settings } = loadPiSettings(ctx.cwd);
		if (settings.betterCompaction?.enabled === false) return;
		const thinking = settings.betterCompaction?.thinking ?? "inherit";
		const noThinking =
			thinking === "off" || (thinking === "inherit" && (!ctx.thinkingLevel || ctx.thinkingLevel === "off"));
		if (noThinking && ctx.hasUI) {
			ctx.ui.notify(
				"Better Compaction: compaction will run with thinking turned off — some models summarize significantly worse (or not at all) without it",
				"warning",
			);
		}
	});

	// /bc — inspect and control this extension's options.
	pi.registerCommand("bc", {
		description:
			"Better Compaction: /bc [on|off|thinking off|inherit|report on|off] — show or change compaction options",
		getArgumentCompletions: (prefix) => {
			const options = ["on", "off", "thinking", "report"];
			const filtered = options.filter((o) => o.startsWith(prefix.toLowerCase()));
			return filtered.length > 0
				? filtered.map((o) => ({ value: o, label: o, description: "Better Compaction option" }))
				: null;
		},
		handler: async (args, ctx) => {
			const { settings, betterCompactionSource, betterCompactionSourcePath } = loadPiSettings(ctx.cwd);
			const section: BetterCompactionSettings = settings.betterCompaction ?? {};
			const enabled = section.enabled !== false;
			const report = section.report !== false;
			const thinking = section.thinking ?? "inherit";
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
				if (value !== "off" && value !== "inherit") {
					ctx.ui.notify("Better Compaction: usage: /bc thinking off|inherit", "error");
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
			ctx.ui.notify(
				"Better Compaction: usage: /bc [on|off|thinking off|inherit|report on|off]",
				"error",
			);
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

		// "inherit" (default) steers the model to keep thinking brief; "off"
		// steers it to skip thinking entirely. In both cases the request keeps
		// the session's thinking mode (see the reasoning block below), so the
		// prompt cache is preserved.
		const thinkingChoice = settings.betterCompaction?.thinking ?? "inherit";

		// 1. Rebuild the prefix of pi's LLM request up to the compaction cut.
		//
		// preparation.firstKeptEntryId marks where pi starts KEEPING messages
		// verbatim; everything before it is what gets summarized and discarded.
		// The agent's last request was [system prompt + full conversation], and
		// the conversation up to the cut is a prefix of that, so prefix/KV
		// caches still cover it. The kept tail is deliberately omitted: the
		// summarizer must not re-describe messages that stay in the session.
		//
		// The prefix is rebuilt from the EDIT-AWARE projection
		// (buildSessionProjection), not the raw entries: the agent's actual LLM
		// context applies context_edit entries (content replacements and
		// omissions, e.g. from context-overflow recovery), and pi's own
		// prepareCompaction works off this same projection. Rebuilding from the
		// raw entries would make the request diverge from the cached agent
		// prefix at the first edited message (silently losing the cache hit)
		// and would feed the summarizer content the model never had. Each
		// projected entry keeps its sourceEntry, so the cut point maps 1:1.
		const entries = ctx.sessionManager.getEntries();
		const projection = buildSessionProjection(entries, ctx.sessionManager.getLeafId());
		const cutIndex = projection.entries.findIndex((e) => e.sourceEntry.id === preparation.firstKeptEntryId);
		if (cutIndex < 1) return; // nothing to summarize — let pi decide
		const prefixMessages = projection.entries.slice(0, cutIndex).flatMap((e) => e.messages);
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
		// Steer thinking in the appended prompt rather than in the request
		// (changing the request's thinking mode would force a full prefill):
		// the summary is short structured output, so we don't want a long
		// thinking phase on it. "off" asks the model to skip thinking entirely;
		// "inherit" (default) asks it to keep thinking brief.
		promptText += thinkingChoice === "off" ? NO_THINKING_INSTRUCTION : BRIEF_THINKING_INSTRUCTION;
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
		// Thinking level for the summarization call: always the session's own
		// level. We never change the thinking mode in the request — with some
		// models the thinking mode is applied at the chat-template level (the
		// template injects different prompts into the running chat), so
		// switching it mid-session changes the prompt structure and forces a
		// full prefill of the whole conversation. Keeping the request identical
		// to the agent's last request is what lets the backend reuse the prompt
		// cache. To make the model think less we steer it in the appended
		// prompt instead (BRIEF_THINKING_INSTRUCTION / NO_THINKING_INSTRUCTION).
		if (model.reasoning && ctx.thinkingLevel && ctx.thinkingLevel !== "off") {
			streamOptions.reasoning = ctx.thinkingLevel;
		}

		// Same agent-level retry policy pi's built-in compaction uses
		// (settings.retry.*), so the compaction call honors retry.enabled too.
		// The stream is consumed manually (same events .result() uses) so the
		// final attempt can be timed: thinking time is the span from
		// thinking_start until the model moves on to a non-thinking block
		// (text_start/toolcall_start) — OpenAI-compatible adapters emit
		// thinking_end only after the stream has fully ended (blocks are
		// finalized in bulk), so on those backends waiting for thinking_end
		// would swallow the whole summary generation into "thinking" — the
		// rest is generation.
		let timing = { totalMs: 0, thinkingMs: 0 };
		const produce = async () => {
			const t0 = Date.now();
			let thinkingMs = 0;
			let spanStart: number | null = null;
			let message: AssistantMessage | undefined;
			const stream = await ctx.modelRegistry.streamSimple(model, requestContext, streamOptions);
			for await (const event of stream) {
				if (event.type === "thinking_start" && spanStart === null) {
					spanStart = Date.now();
				} else if (event.type === "thinking_end" ||
					// Close the span as soon as the answer starts: on
					// OpenAI-compatible adapters thinking_end only arrives at
					// stream end, and text_start is the true end of thinking.
					// (On adapters that emit thinking_end mid-stream, it has
					// already closed the span, so text_start is a no-op.)
					event.type === "text_start" || event.type === "toolcall_start") {
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
		};
		const retryPolicy = {
			enabled: settings.retry?.enabled ?? true,
			maxRetries: settings.retry?.maxRetries ?? 3,
			baseDelayMs: settings.retry?.baseDelayMs ?? 2000,
			maxAgentDelayMs: settings.retry?.maxAgentDelayMs ?? 60_000,
		};
		let response = await retryAssistantCall(produce, retryPolicy, signal);

		// 4. Validate like the built-in compaction does: never persist a
		//    partial or failed summary.
		//
		// The summarized conversation is full of real tool calls, so some
		// models — especially smaller local ones — attempt to call a tool
		// instead of just summarizing, despite the prompt forbidding it.
		// Recover: keep the summary text when there is one, otherwise retry
		// the identical request once before giving up.
		const hasToolCall = (r: AssistantMessage) => r.content.some((b) => b.type === "toolCall");
		const validateResponse = (r: AssistantMessage) => {
			if (r.stopReason === "aborted") throw new Error("Compaction aborted");
			if (r.stopReason === "error") {
				throw new Error(`Summarization failed: ${r.errorMessage || "Unknown error"}`);
			}
			if (r.stopReason === "length" && !hasToolCall(r)) {
				throw new Error("Summarization failed: generation hit the token cap and the summary is incomplete");
			}
		};
		validateResponse(response);
		if (hasToolCall(response) && !contentText(response.content).trim()) {
			response = await retryAssistantCall(produce, retryPolicy, signal);
			validateResponse(response);
		}
		const summaryText = contentText(response.content).trim();
		if (hasToolCall(response) && !summaryText) {
			throw new Error("Summarization attempted to call a tool");
		}
		if (!summaryText) {
			throw new Error("Summarization returned no text");
		}

		// 5. Same cumulative file tracking as the built-in compaction.
		const { readFiles, modifiedFiles } = computeFileLists(preparation.fileOps);
		const summary = summaryText + formatFileOperations(readFiles, modifiedFiles);

		// 6. Stash the report (measured on the final attempt — `timing` is
		//    rewritten by every produce() call, so after the validation/retry
		//    step above it holds the final attempt's numbers, not a partial or
		//    pre-validation value) for the session_compact event. It must NOT
		//    be emitted here: session_before_compact runs before pi applies the
		//    compaction result, so a message appended in this handler would
		//    claim "completed" before the compaction lifecycle is finished.
		pendingReports.set(ctx.sessionManager.getSessionId(), {
			totalMs: timing.totalMs,
			thinkingMs: timing.thinkingMs,
			cacheRead: response.usage.cacheRead,
		});

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

	// pi fires session_compact only after it has applied the compaction result
	// and completed the compaction lifecycle, so this is where the report is
	// actually emitted. A stashed report exists only when THIS extension
	// supplied the compaction — for the built-in compaction (or when the
	// handler returned early) there is nothing stashed, so no report is sent.
	// The report is appended as a custom session entry: it shows in the chat
	// (see the entry renderer above) and stays in the transcript, but is never
	// sent to the LLM. No UI is required — in non-interactive modes the entry
	// is still persisted and renders whenever the session is viewed.
	pi.on("session_compact", (_event, ctx) => {
		const sessionId = ctx.sessionManager.getSessionId();
		const report = pendingReports.get(sessionId);
		if (!report) return; // built-in compaction — not ours to report
		pendingReports.delete(sessionId); // emitted exactly once
		// Read at emit time like every other option: report=false hides it.
		const { settings } = loadPiSettings(ctx.cwd);
		if (settings.betterCompaction?.report === false) return;
		pi.appendEntry<CompactionReportData>("better-compaction-report", {
			totalMs: report.totalMs,
			thinkingMs: report.thinkingMs,
			cacheRead: report.cacheRead,
			timestamp: Date.now(),
		});
	});

	// A failed or aborted compaction must not leave a stale report behind that
	// a later session_compact would emit for the wrong compaction. (A shutdown
	// mid-compaction is safe too: the runtime — and this map — is torn down.)
	pi.on("session_compact_failed", (_event, ctx) => {
		pendingReports.delete(ctx.sessionManager.getSessionId());
	});
}
