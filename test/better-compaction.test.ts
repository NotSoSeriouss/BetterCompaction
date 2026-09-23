/**
 * better-compaction report lifecycle tests.
 *
 * Regression guard for the post-compaction report: it must be appended as a
 * custom session entry (persistent in the chat, never sent to the LLM) on pi's
 * `session_compact` event (which fires only after pi has applied the
 * compaction result), NOT inside `session_before_compact` — and only for
 * compactions this extension actually produced.
 *
 * Run with: bun test
 */
/// <reference types="bun-types" />
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import extension from "../extensions/better-compaction";

// ---------------------------------------------------------------------------
// Mock harness
// ---------------------------------------------------------------------------

interface ApiMock {
	/** event → handler (in registration order) */
	handlers: Map<string, unknown[]>;
	/** entries appended via pi.appendEntry, in order */
	entries: Array<{ customType: string; data?: unknown }>;
	on: (event: string, handler: unknown) => () => void;
	registerCommand: (name: string, def: unknown) => void;
	registerEntryRenderer: (customType: string, renderer: unknown) => void;
	appendEntry: (customType: string, data?: unknown) => void;
}

function createApiMock(): ApiMock {
	const handlers = new Map<string, unknown[]>();
	const entries: Array<{ customType: string; data?: unknown }> = [];
	return {
		handlers,
		entries,
		on: (event, handler) => {
			const list = handlers.get(event) ?? [];
			list.push(handler);
			handlers.set(event, list);
			return () => {
				const i = list.indexOf(handler);
				if (i >= 0) list.splice(i, 1);
			};
		},
		registerCommand: () => {},
		registerEntryRenderer: () => {},
		appendEntry: (customType, data) => {
			entries.push({ customType, data });
		},
	};
}

/** Fire the (single) handler registered for `event` and return its result. */
async function fire(api: ApiMock, event: string, eventPayload: unknown, ctx: unknown) {
	const list = api.handlers.get(event);
	expect(list, `no handler registered for ${event}`).toBeDefined();
	const handler = list![list!.length - 1] as (e: unknown, c: unknown) => unknown;
	return handler(eventPayload, ctx);
}

// ---------------------------------------------------------------------------
// Session entry / stream fixtures
// ---------------------------------------------------------------------------

function makeEntries(): { entries: unknown[]; leafId: string; keptEntryId: string } {
	const mk = (i: number, message: unknown) => ({
		type: "message",
		id: `entry-${i}`,
		parentId: i === 0 ? null : `entry-${i - 1}`,
		timestamp: new Date(1_700_000_000_000 + i * 1000).toISOString(),
		message,
	});
	const user = (text: string) => ({
		role: "user" as const,
		content: text,
		timestamp: Date.now(),
	});
	const assistant = (text: string) => ({
		role: "assistant" as const,
		content: [{ type: "text" as const, text }],
		api: "openai-completions" as const,
		provider: "test",
		model: "test-model",
		usage: { input: 10, output: 5, cacheRead: 0, cacheWrite: 0 },
		stopReason: "stop" as const,
		timestamp: Date.now(),
	});
	const entries = [
		mk(0, user("Read the codebase and tell me what it does.")),
		mk(1, assistant("It is a small CLI tool.")),
		mk(2, user("Now add a config file option.")), // firstKeptEntryId — cut point
		mk(3, assistant("Done, added --config.")),
		mk(4, user("Great, commit it.")),
	];
	return { entries, leafId: "entry-4", keptEntryId: "entry-2" };
}

function makePreparation(keptEntryId: string) {
	return {
		firstKeptEntryId: keptEntryId,
		messagesToSummarize: [] as unknown[],
		turnPrefixMessages: [] as unknown[],
		isSplitTurn: false,
		tokensBefore: 123_456,
		fileOps: { read: new Set<string>(), written: new Set<string>(), edited: new Set<string>() },
		settings: { reserveTokens: 20_000 },
	};
}

function makeSummaryMessage(overrides?: {
	cacheRead?: number;
	stopReason?: "stop" | "error";
	errorMessage?: string;
	text?: string;
}) {
	return {
		role: "assistant" as const,
		content: [{ type: "text" as const, text: overrides?.text ?? "## Goal\nKeep the test green." }],
		api: "openai-completions" as const,
		provider: "test",
		model: "test-model",
		usage: {
			input: 100,
			output: 40,
			cacheRead: overrides?.cacheRead ?? 12_345,
			cacheWrite: 0,
		},
		stopReason: overrides?.stopReason ?? ("stop" as const),
		...(overrides?.errorMessage ? { errorMessage: overrides.errorMessage } : {}),
		timestamp: Date.now(),
	};
}

/** Fake LLM stream with a ~300ms thinking span so the report shows the split. */
function makeStream(message: unknown) {
	const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
	return {
		async *[Symbol.asyncIterator]() {
			yield { type: "thinking_start" };
			await sleep(300);
			yield { type: "thinking_end" };
			yield { type: "done", message };
		},
	};
}

/**
 * Fake LLM stream in the order the OpenAI-compatible adapter actually emits:
 * thinking_start … text_start (answer begins) … then thinking_end ONLY at
 * stream end (blocks are finalized in bulk after the SSE loop).
 */
function makeOpenAiOrderStream(message: unknown) {
	const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
	return {
		async *[Symbol.asyncIterator]() {
			yield { type: "thinking_start" };
			await sleep(300);
			yield { type: "text_start" };
			await sleep(600);
			yield { type: "thinking_end" }; // late: only at stream end
			yield { type: "done", message };
		},
	};
}

function makeCtx(opts: {
	sessionId: string;
	cwd: string;
	notify: (message: string, type?: string) => void;
	/** produce() results, consumed one per streamSimple call */
	messages: unknown[];
	hasUI?: boolean;
	/** stream shape used by the fake streamSimple (default: makeStream) */
	streamFactory?: (message: unknown) => unknown;
}) {
	const { entries, leafId } = makeEntries();
	const calls: unknown[][] = [];
	return {
		calls,
		model: {
			id: "test-model",
			provider: "test",
			maxTokens: 8192,
			reasoning: true,
		},
		modelRegistry: {
			streamSimple: async (...args: unknown[]) => {
				calls.push(args);
				const msg = opts.messages.shift() ?? makeSummaryMessage();
				return (opts.streamFactory ?? makeStream)(msg);
			},
		},
		sessionManager: {
			getEntries: () => entries,
			getLeafId: () => leafId,
			getSessionId: () => opts.sessionId,
		},
		cwd: opts.cwd,
		hasUI: opts.hasUI ?? true,
		thinkingLevel: "off",
		signal: new AbortController().signal,
		ui: { notify: opts.notify },
	};
}

function beforeCompactEvent(keptEntryId: string) {
	return {
		type: "session_before_compact",
		preparation: makePreparation(keptEntryId),
		branchEntries: [],
		reason: "manual",
		willRetry: false,
		signal: new AbortController().signal,
	};
}

function compactEvent() {
	return {
		type: "session_compact",
		compactionEntry: { type: "compaction", id: "c1", parentId: null, timestamp: "" },
		fromExtension: true,
		reason: "manual",
		willRetry: false,
	};
}

function compactFailedEvent() {
	return {
		type: "session_compact_failed",
		reason: "manual",
		errorMessage: "boom",
		aborted: false,
		willRetry: false,
		fromExtension: true,
	};
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

let tmp: string;
let projectDir: string;

beforeAll(() => {
	tmp = mkdtempSync(join(tmpdir(), "bc-test-"));
	// Hermetic pi agent dir (global settings): empty → all defaults.
	process.env.PI_CODING_AGENT_DIR = join(tmp, "agent");
	mkdirSync(process.env.PI_CODING_AGENT_DIR);
	projectDir = join(tmp, "project");
	mkdirSync(projectDir);
});

afterAll(() => {
	rmSync(tmp, { recursive: true, force: true });
});

function freshExtension(): ApiMock {
	const api = createApiMock();
	extension(api as never);
	return api;
}

describe("post-compaction report lifecycle", () => {
	test("is NOT emitted during session_before_compact, only on session_compact", async () => {
		const api = freshExtension();
		const ctx = makeCtx({ sessionId: "s1", cwd: projectDir, notify: () => {}, messages: [] });

		// Summarization runs and the compaction result is returned…
		const result = await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		expect(result).toBeDefined();
		const compaction = (result as { compaction: Record<string, unknown> }).compaction;
		expect(compaction.details).toMatchObject({ strategy: "better-compaction" });
		expect(compaction.firstKeptEntryId).toBe("entry-2");
		expect(compaction.tokensBefore).toBe(123_456);
		expect(String(compaction.summary)).toContain("## Goal");
		// …but nothing is reported yet: the compaction is not complete.
		expect(api.entries).toEqual([]);

		// …and the report entry appears exactly when pi says the compaction
		// succeeded, with the final attempt's numbers.
		await fire(api, "session_compact", compactEvent(), ctx);
		expect(api.entries).toHaveLength(1);
		expect(api.entries[0]!.customType).toBe("better-compaction-report");
		expect(api.entries[0]!.data).toMatchObject({ cacheRead: 12_345 });

		// …and never again for the same compaction.
		await fire(api, "session_compact", compactEvent(), ctx);
		expect(api.entries).toHaveLength(1);
	});

	test("reports the final attempt's timing after a retried summarization", async () => {
		const api = freshExtension();
		const messages: string[] = [];
		const cwd = join(tmp, "retry-project");
		mkdirSync(join(cwd, ".pi"), { recursive: true });
		// Fast retry so the test stays quick.
		writeFileSync(
			join(cwd, ".pi", "settings.json"),
			JSON.stringify({ retry: { enabled: true, maxRetries: 3, baseDelayMs: 10, maxAgentDelayMs: 100 } }),
		);
		// First attempt fails with a retryable error, second succeeds with
		// different (final) usage — the report must reflect the final attempt.
		const ctx = makeCtx({
			sessionId: "s2",
			cwd,
			notify: (m) => messages.push(m),
			messages: [
				makeSummaryMessage({ stopReason: "error", errorMessage: "overloaded, please retry" }),
				makeSummaryMessage({ cacheRead: 55, text: "## Goal\nFinal attempt summary." }),
			],
		});

		const result = await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		expect((result as { compaction: { summary: string } }).compaction.summary).toContain("Final attempt");
		expect(api.entries).toEqual([]);

		await fire(api, "session_compact", compactEvent(), ctx);
		expect(api.entries).toHaveLength(1);
		expect(api.entries[0]!.customType).toBe("better-compaction-report");
		// Must be the final attempt's usage (55), not attempt 1's 12_345.
		expect(api.entries[0]!.data).toMatchObject({ cacheRead: 55 });
	});

	test("does not report for pi's built-in compaction (no stashed report)", async () => {
		const api = freshExtension();
		const ctx = makeCtx({ sessionId: "s3", cwd: projectDir, notify: () => {}, messages: [] });

		// session_compact without a preceding session_before_compact result
		// (i.e. pi's built-in compaction ran) → no report.
		await fire(api, "session_compact", compactEvent(), ctx);
		expect(api.entries).toEqual([]);
	});

	test("discards the stashed report when the compaction fails", async () => {
		const api = freshExtension();
		const ctx = makeCtx({ sessionId: "s4", cwd: projectDir, notify: () => {}, messages: [] });

		// Summarization succeeded (report stashed) but the compaction then fails.
		const result = await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		expect(result).toBeDefined();
		await fire(api, "session_compact_failed", compactFailedEvent(), ctx);
		expect(api.entries).toEqual([]);

		// A later successful compaction in the same session must not emit the
		// stale report from the failed one (built-in path → nothing stashed).
		await fire(api, "session_compact", compactEvent(), ctx);
		expect(api.entries).toEqual([]);
	});

	test("respects report=false (set at emit time)", async () => {
		const api = freshExtension();
		const cwd = join(tmp, "report-off");
		mkdirSync(join(cwd, ".pi"), { recursive: true });
		writeFileSync(join(cwd, ".pi", "settings.json"), JSON.stringify({ betterCompaction: { report: false } }));
		const ctx = makeCtx({ sessionId: "s5", cwd, notify: () => {}, messages: [] });

		const result = await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		expect(result).toBeDefined(); // summary is still produced…
		await fire(api, "session_compact", compactEvent(), ctx);
		expect(api.entries).toEqual([]); // …but the report is suppressed
	});

	test("persists the report entry even when there is no UI", async () => {
		const api = freshExtension();
		const ctx = makeCtx({
			sessionId: "s6",
			cwd: projectDir,
			notify: () => {},
			messages: [],
			hasUI: false,
		});

		const result = await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		expect(result).toBeDefined();
		await fire(api, "session_compact", compactEvent(), ctx);
		// The report is a session entry, not a notification: no UI is needed,
		// and it renders whenever the session is viewed interactively.
		expect(api.entries).toHaveLength(1);
		expect(api.entries[0]!.customType).toBe("better-compaction-report");
	});

	test("keeps reports isolated per session", async () => {
		const api = freshExtension();
		const ctxA = makeCtx({ sessionId: "sA", cwd: projectDir, notify: () => {}, messages: [] });
		const ctxB = makeCtx({ sessionId: "sB", cwd: projectDir, notify: () => {}, messages: [] });

		await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctxA);
		await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctxB);

		// Session A's completion must not consume or duplicate session B's report.
		await fire(api, "session_compact", compactEvent(), ctxA);
		expect(api.entries).toHaveLength(1);

		await fire(api, "session_compact", compactEvent(), ctxB);
		expect(api.entries).toHaveLength(2);
	});

	test("closes the thinking span at text_start (OpenAI adapters emit thinking_end only at stream end)", async () => {
		const api = freshExtension();
		const ctx = makeCtx({
			sessionId: "s7",
			cwd: projectDir,
			notify: () => {},
			messages: [],
			streamFactory: makeOpenAiOrderStream,
		});

		await fire(api, "session_before_compact", beforeCompactEvent("entry-2"), ctx);
		await fire(api, "session_compact", compactEvent(), ctx);
		expect(api.entries).toHaveLength(1);
		const data = api.entries[0]!.data as { totalMs: number; thinkingMs: number };
		// The thinking span must end at text_start (~300ms), not at the late
		// thinking_end (~900ms): the 600ms tail is the answer being generated.
		// (With the old code thinkingMs ≈ 900 and "generating" ≈ 0.)
		expect(data.thinkingMs).toBeLessThan(600);
		expect(data.totalMs - data.thinkingMs).toBeGreaterThan(600);
	});
});
