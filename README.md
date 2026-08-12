# clg — hybrid Claude Code launcher

Run Claude Code with a **real Anthropic Opus main thread** (full 1M context, native
compaction) while **every subagent is served by GPT models** through
[claude-code-proxy](https://github.com/raine/claude-code-proxy) (ChatGPT Pro / Codex
backend). The default shared router schedules each proxy request across every
authenticated account with adaptive per-account concurrency and targeted retries.

Why: proxied GPT models struggle with Claude Code's harness (compaction, orchestration),
and pure-Anthropic burns your subscription on bulk subagent work. `clg` gives you the
best of both — the orchestrator is a genuine Anthropic model, the worker fleet is GPT.

```mermaid
flowchart LR
    CC[Claude Code] -->|ANTHROPIC_BASE_URL| R[shared clg router\n:28765]
    R -->|main chat + compaction\nclaude-opus-5 · OAuth| A[api.anthropic.com]
    R -->|scheduled proxy requests| S[AIMD scheduler\npermits + retry]
    S -->|opus/fable→sol\ninherited/sonnet→terra · haiku→luna| P1[proxy account A\n:18765]
    S -->|most free capacity| P2[proxy account B\n:18766]
    P1 --> X[ChatGPT Pro / Codex]
    P2 --> X
```

## Model routing

| Surface | Model | Served by |
|---|---|---|
| Main chat | `claude-opus-5[1m]` (1M context) | api.anthropic.com (your Claude OAuth) |
| Compaction | `claude-opus-5` | api.anthropic.com |
| Subagents `opus` / `fable` | `gpt-5.6-sol` | proxy |
| Subagents `sonnet` | `gpt-5.6-terra` | proxy |
| Subagents `haiku` | `gpt-5.6-luna` | proxy |
| Subagents with **no model** (inherited) — native *and* custom agents | `gpt-5.6-terra` | proxy |
| Any other model id (`claude-sonnet-5`, `claude-haiku-4-5-…`, `grok-*`, `kimi-*`, …) | its tier, else `gpt-5.6-terra` | proxy |

### Compaction is always Claude

Compaction is the one task Claude is specifically trained for, and its summary
conditions everything that follows in the session — so it is served by
`compact_model` (`claude-opus-5`) **regardless of every other setting**. It is
detected by its system prompt alone, never by the incoming model id, because
when the main chat runs on GPT the compaction request arrives carrying a `gpt-*`
name; the router rewrites it.

`main_chat` (`anthropic` | `gpt`) selects where the main chat goes and does
**not** affect compaction:

```sh
clgctl config set main_chat=gpt              # main chat on GPT…
clgctl config set main_model_env=gpt-5.6-sol # …served by sol
# compaction still goes to claude-opus-5
```

`autocompact_pct` (default `90`) is the context fill percentage at which the CLI
auto-compacts — raise it to compact later and less often, lower it to compact
sooner. It is exported as `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`, after that variable
is purged from the inherited environment, so the threshold is always the one
configured here rather than whatever a shell happened to export.

Putting the main chat on GPT costs the 1M window, so compaction stops being
theoretical and starts firing regularly. To keep it from becoming a hard
dependency on Anthropic quota, a compaction that is rate-limited (`429`) or hits
an overloaded API (`529`) retries Anthropic three times and only then degrades to
GPT — logged as `compaction degradee (…)`, because it is a degradation, not a
mode. Auth failures (`401`) are never masked this way: hiding them behind a
fallback would make them undebuggable.

The main chat is, apart from compaction, the **only** surface that reaches Anthropic. Everything else is
served by a GPT model, and model translation is fail-closed: only ids already
prefixed `gpt-` cross the router untouched, everything else is mapped to a tier by
family (`opus`/`fable`→sol, `sonnet`→terra, `haiku`→luna, unknown→inherited). This
matters because the proxy also serves non-GPT models (grok, kimi, deepseek, glm,
qwen): forwarding an id verbatim would hand model choice to the proxy, outside
`routing.json`.

The router tells the main thread apart from inherited-model subagents by the
request's system prompt (whitelist, fail-safe: anything unrecognized or malformed goes
to GPT, never to your Anthropic quota). It accepts one exact trailing `[1m]` suffix only.
Every request is logged as a `route …` line, including the chosen upstream and attempt
number, in `~/.local/state/claude-code-proxy/clg-router-shared.log`.

The `[1m]` suffix matters: without it Claude Code clamps non-first-party auth to a 200k
window and compacts constantly. Opus 5 is natively 1M and the API serves it over
subscription OAuth.

## Requirements

- Claude Code with an active Claude subscription (`claude` logged in — the router reads
  the OAuth token from the macOS Keychain / `~/.claude/.credentials.json` at request time)
- [claude-code-proxy](https://github.com/raine/claude-code-proxy) with at least one
  authenticated Codex account
- Python 3.10+

## Install

```sh
git clone https://github.com/momomuchu/clg && cd clg
./install.sh   # symlinks bin/* into ~/.local/bin
```

## Usage

```sh
clg                    # shared router across every authenticated account
clg @auto              # same shared-router behavior explicitly
clg @b                 # pin account "b" for debugging
clg @list              # accounts + proxy + router + Anthropic OAuth status
clg-account add b      # add a ChatGPT Pro account (device-code login)
clg-fleet -j 3 p1.txt p2.txt   # batch `clg -p` runs
```

Each account gets its own proxy instance (isolated `CCP_CONFIG_DIR`). The default shared
router listens on port `28765`, starts every authenticated account proxy, and picks the
upstream with the most free capacity for each generation request. Each upstream starts
with 6 permits. A WebSocket-upgrade 403 reduces that upstream's permits by 25% (floor 2)
and is retried only when a different account has capacity. Every 20 consecutive completed
generation responses adds one permit (ceiling 12, without lowering explicitly higher
settings). `POST /v1/messages/count_tokens`, `GET /v1/models`, and other non-generation
requests bypass generation permits and AIMD. Requests are capped at 64 MiB and socket
reads time out after 600 seconds. Main-chat Anthropic traffic bypasses this scheduler.

`clg @name` preserves a pinned-session URL on `account port + 10000`, but that listener
is a compatibility relay to the shared router: it pins proxy traffic to `@name` while the
shared router remains the sole owner of permits and AIMD state.

`GET /__clg` reports the configured upstream set plus each upstream's permits, in-flight
count, free capacity, and current success streak.

## Fleet: several Claude Code and Codex accounts

`clgctl` manages three authentication surfaces from one registry
(`~/.config/clg/accounts.json`). Each account lives in its own directory and is logged in
**once**; the credentials stay there and are never asked for again.

| Surface | Isolated by | Added with |
|---|---|---|
| `proxy` — a claude-code-proxy upstream | `CCP_CONFIG_DIR` | `clg-account add <name>` |
| `claude` — a Claude Code config dir | `CLAUDE_CONFIG_DIR` | `clgctl account add --kind claude <name>` |
| `codex` — a Codex CLI home | `CODEX_HOME` | `clgctl account add --kind codex <name>` |

```sh
clgctl status                      # auth + expiry across the whole fleet
clgctl status --deep               # probe each account for real (slower)
clgctl account add --kind claude pro1
clgctl refresh --on all            # refresh tokens that are close to expiring
clgctl run --on all -- codex login status
```

The canonical `~/.claude` and `~/.codex` accounts appear as `@main` and `@codex`. They are
never stored in the registry, moved, or modified.

Enrolment refuses to register an account whose login produced no credentials, and warns
loudly if no new keychain entry appeared — that would mean the login overwrote the
canonical account instead of creating its own store.

### Model routing as configuration

The tier map lives in `~/.config/clg/fleet/gpt/routing.json`. While that file is absent or
unreadable, the values compiled into `bin/clg` apply, so routing is unchanged until you
choose to change it.

```sh
clgctl config show                            # effective routing
clgctl config diff                            # what differs from the built-in defaults
clgctl config set tiers.haiku=gpt-5.6-terra   # fleet-wide
clgctl config set tiers.haiku=gpt-5.6-sol --account b   # this account only
clgctl config unset tiers.haiku --account b   # back to the default
clgctl config apply --dry-run                 # print, write nothing
```

Per-account overrides resolve at launch, in the environment handed to Claude Code — the
shared router serves several accounts over one socket and cannot attribute a request to
one. The router keeps translating aliases (`opus`, `fable`) with the base tiers.

## Caveats

- If the Claude OAuth token expires (long stretches without a normal `claude` session),
  main-chat requests fail — `clg` warns at launch; run `claude` once to refresh.
- A `fork`-type subagent inherits the parent model verbatim and its system prompt looks
  like the main thread's, so it may reach Anthropic.
- The system-prompt whitelist tracks Claude Code's internal prompts; if a CLI update
  rewords them, traffic falls back to GPT (visible in the router log), never the other
  way around.

## License

MIT
