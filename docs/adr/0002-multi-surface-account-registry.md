# ADR-0002: One registry for three authentication surfaces

Status: accepted (2026-08-06)

## Context

`clg` already ran several ChatGPT accounts as proxy upstreams, each isolated by
`CCP_CONFIG_DIR` and given its own port. The Claude Code and Codex CLI sides had no
equivalent: one account each, and switching would have meant logging in again every time.
There was also no way to change a setting once and have it apply everywhere — the model
routing (`claude-opus-5`, `gpt-5.6-sol/terra/luna`) was hardcoded in `bin/clg`.

Three isolation mechanisms were verified on this machine before anything was designed:

- `CLAUDE_CONFIG_DIR` — a fresh directory reports `Not logged in` on a real headless turn,
  and `.claude.json` is written **inside** that directory (it sits **beside** `~/.claude`
  for the default account).
- `CODEX_HOME` — a fresh home reports `Not logged in` while the canonical home reports
  `Logged in using ChatGPT`.
- Claude Code derives its keychain service as `Claude Code-credentials` plus
  `-sha256(<config dir>)[:8]`, with no suffix when `CLAUDE_CONFIG_DIR` is unset. Read out
  of the 2.1.222 bundle, which also exposes `CLAUDE_SECURESTORAGE_CONFIG_DIR` as a way to
  deliberately *share* one credential store.

## Decision

One registry, `~/.config/clg/accounts.json`, gains a `kind` field: `proxy`, `claude` or
`codex`. A missing `kind` means `proxy`, so registries written before this change keep
working untouched. `bin/clgctl` owns the two new surfaces and the fleet-wide operations
(`status`, `refresh`, `run`, `config`); `clg` and `clg-account` keep their existing roles.

Reads are filtered by surface, writes always go through the full registry. The canonical
`~/.claude` and `~/.codex` accounts are shown but never stored, moved or modified.

Model routing moves to `~/.config/clg/fleet/gpt/routing.json`, with the previous hardcoded
values as the built-in defaults, so an absent or corrupt file reproduces today's behaviour
exactly.

## Rationale

The three surfaces already isolate themselves properly; what was missing was a place to
record which account is which, and a single command to act on all of them. Adding a
discriminator to the existing registry costs one field and no migration, where a second
registry would have to be kept in sync with the first.

Per-account routing overrides resolve at **launch**, in the child environment, not in the
router. The shared router serves several accounts over one socket and cannot attribute a
request to an account; the environment handed to Claude Code is what actually selects the
model. The router keeps translating aliases (`opus`, `fable`) with the base tiers.

## Consequences

- `load_registry()` filters to `proxy` by default in `clg` and `clg-account`. A non-proxy
  entry can never reach the upstream scheduler, where it would have been sent to a dead
  port — locked by `tests/test_fleet.py`.
- Writing a filtered view back to disk would delete the other surfaces. `clg-account` and
  `clgctl` therefore load the full registry for mutation; two tests assert that adding an
  account of one kind preserves the others.
- Enrolment refuses to register an account whose login did not produce credentials, and
  warns when no new keychain entry appeared — the signal that a login overwrote the
  canonical account's credentials instead of creating its own.
- `claude auth status` is **not** used as the per-account oracle: it reports
  `loggedIn: true` even inside a never-logged-in config directory. The oracle is the
  presence of `oauthAccount` in that directory's `.claude.json`, with a real headless turn
  behind `status --deep`.
- Codex expiry is read from `access_token`, not `id_token`. The identity token expires
  within hours and stays expired on a perfectly working account; reading it made `status`
  report a healthy account as `EXPIRÉ`.

## Alternatives considered

- **A second registry per surface.** Rejected: three files to keep consistent, and the
  proxy accounts already live in the first one.
- **Deriving the keychain suffix and reading credentials directly.** Rejected as the
  primary mechanism: the derivation is an implementation detail of a bundled binary. It is
  computed as a *hint* (both the raw and the resolved path, since `/tmp` is a symlink to
  `/private/tmp`), but the authoritative value is the entry observed appearing at enrolment.
- **Propagating settings files (`settings.json`, `config.toml`, rules) too.** Out of scope
  by decision: only GPT model routing is propagated. The merge engine is not written, so
  adding a surface later stays a deliberate design step rather than an accidental one.
