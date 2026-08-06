# ADR-0001: Pin sessions through the shared scheduler

Status: accepted (2026-08-06)

## Context

`clg @name` previously launched a dedicated router with an independent scheduler. When
that account was also present in the shared router, both schedulers could exceed the
same upstream account's real concurrency capacity.

## Decision

Start and retain the shared router for `@name`. The historical per-account listener is a
compatibility relay only: it forwards the session to the shared router with a local,
private upstream-selection header. The shared router owns the only permit and AIMD state.

## Rationale

The requested behavior is that `clg @b` sends that session's proxy traffic to `@b`, while
all traffic for `@b` must share one concurrency budget. A delegating listener preserves
the launcher URL contract without a second scheduler.

## Alternatives considered

- Give each `@name` router its own lower permit cap. Rejected because independent caps
  still cannot account for concurrent shared-router traffic.
- Change Claude Code to add a routing header. Rejected because the launcher does not
  control its request headers.

## Consequences

A pinned launcher starts the shared router and all authenticated proxy upstreams. The
compatibility listener does not own permits or AIMD state.
