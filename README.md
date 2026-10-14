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
and to **about 2 minutes** with compaction steered to skip thinking
(`/bc thinking off`).

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
| `/bc thinking off\|inherit` | Set the summarization thinking level |
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
    "thinking": "inherit"
  }
}
```

| Option | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | `false` → the extension stays silent and pi's built-in compaction runs instead |
| `report` | `true` | `false` → don't append the "Compaction completed in Xs ..." message to the chat after a compaction (toggle with `/bc report off`) |
| `thinking` | `"inherit"` | Thinking level for the summarization call: `"inherit"` (the model keeps thinking, but the prompt asks it to keep thinking brief), `"off"` (the prompt asks it to skip thinking entirely and go straight to the summary). In both cases the request itself always runs at the session's own thinking level — see below |

### How thinking is handled (and why the thinking mode is never changed)

Summaries are short structured output, but a reasoning model with thinking
enabled will still burn a long thinking phase on them — on local backends the
thinking phase dominates compaction time. The old design answered this with a
`"low"` thinking level plus a max-thinking-tokens cap. That design is gone, for
a cache reason:

With some models the thinking mode is applied at the **chat-template level**.
The template injects different prompts into the running chat depending on the
mode, so switching the thinking mode (or disabling thinking) mid-session
changes the prompt structure — and that forces the backend to **re-prefill the
whole conversation**, exactly the cost this extension exists to avoid.

So the summarization request always runs at the **session's own thinking
level**, identical to the agent's last request, and the prompt cache is reused
instead of being thrown away. To make the model think less, the extension
steers it in the appended compaction prompt instead:

- `"inherit"` (default) appends a short instruction telling the model to keep
  its thinking brief (this is what the `"low"` cap used to achieve).
- `"off"` appends a stronger instruction telling the model that thinking is
  disabled for this task and to stop thinking immediately and go straight to
  writing the summary. The request still technically carries the session's
  thinking level (to keep the cache), but the model is told not to use it.

This is why there is no longer any way to change the request's thinking mode:
every path keeps it identical to the session's.

### Notifications

- **After each successful compaction** (disable with `/bc report off`):
  `Compaction completed in 42.3s (35.1s thinking, 7.2s generating, 94.4k tokens from cache).`
  The thinking breakdown is shown only when the model actually thought; the
  cache figure shows how many tokens the prefix cache served. The message is
  appended to the chat on pi's `session_compact` event — which fires only
  after pi has applied the compaction result and finished the compaction
  lifecycle — with the timing measured on the final summarization attempt. It
  is stored as a custom session entry, so it stays in the transcript (and
  survives session reloads) but is **never sent to the LLM**. It is *not*
  appended for pi's built-in compaction (the report is stashed per session
  during `session_before_compact` and only exists when this extension
  supplied the summary; a failed compaction discards the stashed report).
- **Once per session, when compaction will run with thinking turned off:**
  `Better Compaction: compaction will run with thinking turned off — some models
  summarize significantly worse (or not at all) without it`

The extension also mirrors the settings pi applies to normal agent requests
(`images.blockImages`, `retry.*`, provider timeouts, transport, thinking
budgets), reading the same settings files pi merges, so the summarization
request stays identical to what the agent sends — including its thinking mode,
which is the key to reusing the prompt cache (see above).

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
