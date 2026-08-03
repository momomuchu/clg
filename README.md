# clg — hybrid Claude Code launcher

Run Claude Code with a **real Anthropic Opus main thread** (full 1M context, native
compaction) while **every subagent is served by GPT models** through
[claude-code-proxy](https://github.com/raine/claude-code-proxy) (ChatGPT Pro / Codex
backend). Multi-account round-robin included.

Why: proxied GPT models struggle with Claude Code's harness (compaction, orchestration),
and pure-Anthropic burns your subscription on bulk subagent work. `clg` gives you the
best of both — the orchestrator is a genuine Anthropic model, the worker fleet is GPT.

```mermaid
flowchart LR
    CC[Claude Code] -->|ANTHROPIC_BASE_URL| R[clg router\n:28765]
    R -->|main chat + compaction\nclaude-opus-5 · OAuth| A[api.anthropic.com]
    R -->|subagents: opus/fable→sol\nsonnet→terra · haiku→luna| P[claude-code-proxy\n:18765]
    P --> X[ChatGPT Pro / Codex]
```

## Model routing

| Surface | Model | Served by |
|---|---|---|
| Main chat | `claude-opus-5[1m]` (1M context) | api.anthropic.com (your Claude OAuth) |
| Compaction | `claude-opus-5` | api.anthropic.com |
| Subagents `opus` / `fable` | `gpt-5.6-sol` | proxy |
| Subagents `sonnet` | `gpt-5.6-terra` | proxy |
| Subagents `haiku` | `gpt-5.6-luna` | proxy |
| Subagents with **no model** (inherited) — native *and* custom agents | `gpt-5.6-sol` | proxy |

The per-account router tells the main thread apart from inherited-model subagents by the
request's system prompt (whitelist, fail-safe: anything unrecognized goes to GPT, never
to your Anthropic quota). Every request is logged as a `route …` line in
`~/.local/state/claude-code-proxy/clg-router-<account>.log`.

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
clg                    # single account, or round-robin if several
clg @b                 # pin account "b"
clg @list              # accounts + proxy + router + Anthropic OAuth status
clg-account add b      # add a ChatGPT Pro account (device-code login)
clg-fleet -j 3 p1.txt p2.txt   # batch `clg -p` runs with retry on the Codex 403 cap
```

Each account gets its own proxy instance (isolated `CCP_CONFIG_DIR`) and its own
router, so the Codex per-account concurrent-session cap is multiplied by the number
of accounts.

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
