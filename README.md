# Better Compaction

Context compaction for [pi](https://pi.dev) that reuses the backend's prompt
cache: instead of summarizing the conversation in a throwaway chat, it
appends the compaction instruction to the conversation itself, so the model
re-reads almost nothing.

## Why

When an AI agent (or an AI harness) needs to compact a conversation, it
usually does this:

1. Take the entire conversation.
2. Start a **new** chat with a system prompt explaining how to compact it.
3. Put the entire conversation below that prompt and ask the model to
   produce the summary.

That works, but it has two problems.

**It throws the cache away.** The summarization chat shares nothing with the
chat that was just running, so the backend's KV/prompt cache is useless: the
model has to pre-fill the *entire* conversation from scratch just to produce
a summary. And compaction happens precisely when the conversation is already
very long — so that full re-read takes a long time, every single time.

**The instruction sits at the top.** Some research has found that, for some
models, an instruction placed at the *end* of a long context performs better
than one placed at the beginning.

Better Compaction flips the design: the summarization request is the
conversation the agent was *already having* — truncated at the compaction cut
point, sent as real messages — with the compaction instruction appended as
the **final message**. The prefix is byte-identical to what the backend
already computed and cached, so it is reused and only the short appended
instruction has to be pre-filled. (On one ~94k-token session, the
summarization request reported 94.4k tokens served from cache.)

The "kept tail" — the most recent messages that pi retains verbatim after
compaction — is deliberately not sent, so the summary doesn't re-describe
messages that remain in the session.

Everything else works exactly like pi's normal compaction: same summary
format, same bookkeeping, same file tracking, same events, same kept-messages
rebuild. The extension only supplies the summary through the
`session_before_compact` event; pi does the rest with its normal path.

## Benchmarks

Better Compaction was developed with **Qwen3.8-27B** in mind (local
llama.cpp deployment), but it works with any model and any backend that
supports prefix/KV caching.

There are no formal benchmarks — compaction time depends heavily on your
hardware and your model. On my machine, with Qwen3.8-27B, the same session
went from **about 13–15 minutes** of compaction down to **about 6.5 minutes**,
and to **about 2 minutes** with thinking disabled (`/bc thinking off`).

## Install

### Requirements

The extension requires **pi** `0.87.1` or [higher](https://github.com/earendil-works/pi/releases).

### Option A — single file (simplest)

Copy the extension file into your pi extensions directory:

```bash
cp extensions/better-compaction.ts ~/.pi/agent/extensions/
```

Restart pi (or start a new session). Done.

### Option B — as a pi package (local path)

```bash
pi install /absolute/path/to/BetterCompaction
```

### Option C — as a pi package (git)

```bash
pi install git:github.com/NotSoSeriouss/BetterCompaction
pi update --extensions   # reconcile to the pinned ref later
```

## Usage

### The `/bc` command

| Command | Effect |
| --- | --- |
| `/bc` | Show effective options: enabled or not, compaction thinking level, session thinking level, and which settings file provides the config |
| `/bc on` | Enable the extension (default) |
| `/bc off` | Disable the extension — pi's built-in compaction takes over |
| `/bc thinking low\|off\|inherit` | Set the summarization thinking level |
| `/bc lowbudget <tokens>` | Set the max thinking tokens sent when compaction runs at "low" (0 = no cap) |
| `/bc report on\|off` | Show / hide the post-compaction report message (timing and cache read) |

`/bc` writes to the settings file that **currently provides** the
`betterCompaction` section — the project file (`.pi/settings.json`) if it
defines one, otherwise the global file (`~/.pi/agent/settings.json`) — so it
always edits the file that is actually in effect. Changes are read at
compaction time, so they apply at the **next compaction** without restarting.

### Settings

`~/.pi/agent/settings.json` (global) or `.pi/settings.json` (project, wins):

```json
{
  "betterCompaction": {
    "enabled": true,
    "report": true,
    "thinking": "low",
    "lowThinkingBudget": 2048,
    "thinkingBudgetField": "auto"
  }
}
```

| Option | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | `false` → the extension stays silent and pi's built-in compaction runs instead |
| `report` | `true` | `false` → don't send the "Compaction completed in Xs ..." message after a compaction (toggle with `/bc report off`) |
| `thinking` | `"low"` | Thinking level for the summarization call: `"low"` (short structured summaries rarely need more, and a high level can hit the model's thinking cap mid-summary), `"off"` (omits the reasoning parameter — identical to the agent's own requests when session thinking is off), `"inherit"` (follow the session's level — the built-in behavior) |
| `lowThinkingBudget` | `2048` | Max thinking tokens sent when the summarization runs at thinking level `"low"` (also with `"inherit"` when the session level is low). Falls back to the global `thinkingBudgets.low`, then 2048. `0` disables the cap (toggle with `/bc lowbudget`) |
| `thinkingBudgetField` | `"auto"` | The top-level OpenAI-compatible request field that carries the cap: `"thinking_budget_tokens"` (llama.cpp), `"thinking_token_budget"` (vLLM), `"thinking_budget"` (Qwen/DashScope/SGLang). `"auto"` uses the model's `compat.thinkingTokenBudgetField` when set, otherwise the llama.cpp field |

### How the "low" thinking cap works

Summaries are short structured output, but a reasoning model with thinking
enabled will still burn a long thinking phase on them — on local backends the
thinking phase dominates compaction time and can hit the model's thinking cap
mid-summary. So when the summarization call runs at thinking level `"low"`
(the default), the extension also sends a max-thinking-tokens cap as a
top-level field of the OpenAI-compatible request body, e.g.
`"thinking_budget_tokens": 2048` for a local llama.cpp server. The cap is
injected through pi-ai's `samplingParams` pass-through, which merges the field
into the request body last (so it wins over any model-level value) and only
OpenAI-compatible adapters apply — calls on other APIs (Anthropic, ...) are
unchanged.

The value is `lowThinkingBudget` (default 2048, falling back to the global
`thinkingBudgets.low`), clamped so at least 1024 tokens remain for the summary
itself under the response ceiling. Note that llama.cpp only honors the
per-request field when the server was started without `--reasoning-budget`
(a server-level budget always wins).

### Notifications

- **After each successful compaction** (disable with `/bc report off`):
  `Compaction completed in 42.3s (35.1s thinking, 7.2s generating, 94.4k tokens from cache).`
  The thinking breakdown is shown only when the model actually thought; the
  cache figure shows how many tokens the prefix cache served. The message is
  emitted on pi's `session_compact` event — which fires only after pi has
  applied the compaction result and finished the compaction lifecycle — with
  the timing measured on the final summarization attempt. It is *not* sent for
  pi's built-in compaction (the report is stashed per session during
  `session_before_compact` and only exists when this extension supplied the
  summary; a failed compaction discards the stashed report).
- **Once per session, when compaction will run without thinking:**
  `Better Compaction: thinking is disabled for compaction — some models
  summarize significantly worse (or not at all) without it`

The extension also mirrors the settings pi applies to normal agent requests
(`images.blockImages`, `retry.*`, provider timeouts, transport, thinking
budgets), reading the same settings files pi merges, so the summarization
request stays identical to what the agent sends — plus the "low" thinking
cap described above, which the agent's own requests don't carry.

## Repository layout

```
better-compaction/
├── package.json               # pi package manifest (pi.extensions, pi-package keyword)
├── README.md
├── .gitignore
└── extensions/
    └── better-compaction.ts   # the extension (single file, no build step)
```

- `package.json` declares `"pi": { "extensions": ["./extensions"] }` and the
  `pi-package` keyword, so `pi install` (local path or git) loads
  `extensions/*.ts` automatically.
- The pi runtime packages (`@earendil-works/pi-coding-agent`,
  `@earendil-works/pi-ai`) are `peerDependencies` with `*` ranges: they are
  provided by pi itself and are never bundled.
- There are no other dependencies and no build step — the extension is plain
  TypeScript loaded by pi's jiti loader.

## Development notes

- Type-checking locally: pi resolves the peer packages with its own loader
  aliases at runtime. To type-check outside pi, point a `tsconfig.json`
  `paths` entry for `@earendil-works/pi-coding-agent` and
  `@earendil-works/pi-ai` at the installed pi package
  (`<pi-install>/dist/index.d.ts` and
  `<pi-install>/node_modules/@earendil-works/pi-ai/dist/index.d.ts`).
- Known limitations (documented, intentional):
  - If other extensions or pi features transform the LLM context
    (`transformContext` handlers, forced prompts), the rebuilt prefix may no
    longer be byte-identical to the agent's actual request; cache reuse then
    degrades gracefully to a shorter shared prefix.
  - Provider attribution headers (telemetry) that pi adds for a few specific
    providers are not replicated; they don't affect prompt caching.
  - File tracking has the same `fromHook` gap as the built-in compaction.

## License

MIT
