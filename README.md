# Better Compaction

Cache-friendly context compaction for [pi](https://pi.dev).

When pi's context fills up, its built-in compaction asks the model to summarize
the conversation using a **one-off request that looks nothing like the
conversation itself**:

```
[summarization system prompt] + [whole conversation serialized as ONE user message]
```

That request shares no prefix with the last normal agent request, so the
backend has to prefill the entire conversation **from scratch** to produce the
summary. On large sessions this is slow and expensive.

Better Compaction replaces that request with:

```
[the agent's own request, truncated at the compaction cut point —
 system prompt + conversation up to the first kept entry, as real messages]
+
[ONE user message at the very end containing the compaction prompt]
```

Because the prefix is byte-identical to a prefix of the previous agent
request, any backend with prefix/KV caching (llama.cpp `llama-server`,
Anthropic, OpenAI, ...) **reuses the cached prefix** and only prefills the
small appended prompt. Measured on a ~94k-token session: the summarization
request reported `cacheRead: 94.4k tokens` — the entire conversation came
from cache.

It also cuts the conversation at the compaction cut point: the "kept tail"
(the recent messages pi retains verbatim) is **not** sent to the summarizer,
so the summary doesn't re-describe messages that remain in the session.

Everything else works exactly like pi's normal compaction: same summary
format, same `firstKeptEntryId` / `tokensBefore` bookkeeping, same file
tracking, same `session_compact` events, same kept-messages rebuild. The
extension only supplies the summary through the `session_before_compact`
event; pi does the rest with its normal path.

## Install

### Option A — single file (simplest)

Copy the extension file into your pi extensions directory:

```bash
cp extensions/better-compaction.ts ~/.pi/agent/extensions/
```

Restart pi (or start a new session). Done.

### Option B — as a pi package (local path)

This repository is a [pi package](https://pi.dev/packages): `package.json`
declares the `pi` manifest and the `pi-package` keyword, so pi can install
the whole directory directly:

```bash
pi install /absolute/path/to/better-compaction
```

This writes the local path into your user settings (`~/.pi/agent/settings.json`).
Use `pi install -l ...` to install into project settings (`.pi/settings.json`)
instead. Remove with `pi remove /absolute/path/to/better-compaction`.

### Option C — as a pi package (git)

Push this repository to a git host and install it by URL:

```bash
pi install git:github.com/<you>/better-compaction@v1
pi update --extensions   # reconcile to the pinned ref later
```

> **Note:** don't install the package *and* keep a copy of the file in
> `~/.pi/agent/extensions/` at the same time — the extension would load twice.
> Pick one.

## Usage

### The `/bc` command

| Command | Effect |
| --- | --- |
| `/bc` | Show effective options: enabled or not, compaction thinking level, session thinking level, and which settings file provides the config |
| `/bc on` | Enable the extension (default) |
| `/bc off` | Disable the extension — pi's built-in compaction takes over |
| `/bc thinking low\|off\|inherit` | Set the summarization thinking level |

`/bc` writes to the settings file that **currently provides** the
`appendCompaction` section — the project file (`.pi/settings.json`) if it
defines one, otherwise the global file (`~/.pi/agent/settings.json`) — so it
always edits the file that is actually in effect. Changes are read at
compaction time, so they apply at the **next compaction** without restarting.

### Settings

`~/.pi/agent/settings.json` (global) or `.pi/settings.json` (project, wins):

```json
{
  "appendCompaction": {
    "enabled": true,
    "thinking": "low"
  }
}
```

| Option | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | `false` → the extension stays silent and pi's built-in compaction runs instead |
| `thinking` | `"low"` | Thinking level for the summarization call: `"low"` (short structured summaries rarely need more, and a high level can hit the model's thinking cap mid-summary), `"off"` (omits the reasoning parameter — identical to the agent's own requests when session thinking is off), `"inherit"` (follow the session's level — the built-in behavior) |

### Notifications

- **After each successful compaction:**
  `Compaction completed in 42.3s (35.1s thinking, 7.2s generating, 94.4k tokens from cache).`
  The thinking breakdown is shown only when the model actually thought; the
  cache figure shows how many tokens the prefix cache served.
- **Once per session, when compaction will run without thinking:**
  `Better Compaction: thinking is disabled for compaction — some models
  summarize significantly worse (or not at all) without it`

The extension also mirrors the settings pi applies to normal agent requests
(`images.blockImages`, `retry.*`, provider timeouts, transport, thinking
budgets), reading the same settings files pi merges, so the summarization
request stays identical to what the agent sends.

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
