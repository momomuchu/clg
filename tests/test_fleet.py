#!/usr/bin/env python3
"""Behavior locks for the multi-surface account registry and the fleet tools.

The registry at ~/.config/clg/accounts.json holds three kinds of account:
`proxy` (a claude-code-proxy upstream), `claude` (a Claude Code config dir) and
`codex` (a Codex CLI home). Only `proxy` entries may ever reach the router's
upstream list — anything else would be scheduled as a GPT backend it is not.

These tests exist because that separation is invisible at runtime: a misfiled
entry does not crash, it silently routes traffic to a dead port.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import io
import json
import os
import pathlib
import re
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout


BIN = pathlib.Path(__file__).parents[1] / "bin"


def load_script(path: pathlib.Path, name: str) -> types.ModuleType:
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


clg = load_script(BIN / "clg", "clg_launcher_fleet")
clg_account = load_script(BIN / "clg-account", "clg_account")
clgctl = load_script(BIN / "clgctl", "clgctl")


LEGACY_REGISTRY = {
    "a": {"dir": "/tmp/ccp-a", "port": 18765, "brew": True},
    "b": {"dir": "/tmp/ccp-b", "port": 18766},
}

MIXED_REGISTRY = {
    "a": {"kind": "proxy", "dir": "/tmp/ccp-a", "port": 18765, "brew": True},
    "b": {"kind": "proxy", "dir": "/tmp/ccp-b", "port": 18766},
    "perso": {"kind": "claude", "home": "/tmp/claude-perso", "canonical": True},
    "pro1": {"kind": "claude", "home": "/tmp/claude-pro1"},
    "cdx1": {"kind": "codex", "home": "/tmp/codex-cdx1"},
}


class Fixture(unittest.TestCase):
    """Point every module's REGISTRY / ROUTING_FILE at scratch paths."""

    def scratch(self) -> pathlib.Path:
        return pathlib.Path(tempfile.mkdtemp(prefix="clg-test-"))

    def use_registry(self, payload: dict) -> pathlib.Path:
        root = self.scratch()
        path = root / "accounts.json"
        path.write_text(json.dumps(payload))
        for module in (clg, clg_account, clgctl):
            self.patch(module, "REGISTRY", path)
        self.patch(clgctl, "CFG_DIR", root)
        return path

    def use_routing(self, payload: object | None) -> pathlib.Path:
        root = self.scratch()
        path = root / "routing.json"
        if payload is not None:
            path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
        for module in (clg, clgctl):
            self.patch(module, "ROUTING_FILE", path)
        return path

    def patch(self, module: types.ModuleType, attr: str, value: object) -> None:
        original = getattr(module, attr)
        setattr(module, attr, value)
        self.addCleanup(setattr, module, attr, original)


# ---------------------------------------------------------------------------
# Registre
# ---------------------------------------------------------------------------

class TestProxyIsolation(Fixture):
    def test_legacy_registry_is_unchanged_for_the_router(self):
        """A registry written before `kind` existed must route exactly as before."""
        self.use_registry(LEGACY_REGISTRY)
        self.assertEqual(sorted(clg.load_registry()), ["a", "b"])
        self.assertEqual(clg.load_registry()["b"]["port"], 18766)

    def test_non_proxy_accounts_never_become_upstreams(self):
        self.use_registry(MIXED_REGISTRY)
        proxies = clg.load_registry()
        self.assertEqual(sorted(proxies), ["a", "b"])
        # has_auth() indexes acc["dir"]; a claude/codex entry has no "dir" and
        # would raise. Proving the filter runs first is the point of this test.
        for acc in proxies.values():
            self.assertIn("dir", acc)

    def test_full_registry_is_reachable_when_asked(self):
        self.use_registry(MIXED_REGISTRY)
        self.assertEqual(sorted(clg.load_registry(kind=None)), sorted(MIXED_REGISTRY))

    def test_next_proxy_port_ignores_other_surfaces(self):
        """A claude entry has no port; next_port() must not see it."""
        self.use_registry(MIXED_REGISTRY)
        self.assertEqual(clg_account.next_port(clg_account.load_registry()), 18767)


class TestRegistryWritesAreNonDestructive(Fixture):
    def test_adding_a_proxy_account_preserves_claude_and_codex_entries(self):
        """The data-loss trap: mutate the FULL registry, never the filtered view."""
        path = self.use_registry(MIXED_REGISTRY)
        reg = clg_account.load_registry()
        reg["c"] = {"kind": "proxy", "dir": "/tmp/ccp-c", "port": 18767}
        clg_account.save_registry(reg)
        saved = json.loads(path.read_text())
        self.assertEqual(sorted(saved), ["a", "b", "c", "cdx1", "perso", "pro1"])
        self.assertEqual(saved["pro1"]["kind"], "claude")

    def test_clgctl_write_preserves_proxy_entries(self):
        path = self.use_registry(MIXED_REGISTRY)
        reg = clgctl.load_registry()
        reg["pro2"] = {"kind": "claude", "home": "/tmp/claude-pro2"}
        clgctl.save_registry(reg)
        saved = json.loads(path.read_text())
        self.assertEqual(saved["a"]["port"], 18765)
        self.assertEqual(sorted(saved), ["a", "b", "cdx1", "perso", "pro1", "pro2"])

    def test_legacy_entries_gain_an_explicit_kind_on_write(self):
        path = self.use_registry(LEGACY_REGISTRY)
        clgctl.save_registry(clgctl.load_registry())
        saved = json.loads(path.read_text())
        self.assertEqual(saved["a"]["kind"], "proxy")
        self.assertEqual(saved["b"]["kind"], "proxy")


# ---------------------------------------------------------------------------
# Isolation de l'environnement par compte
# ---------------------------------------------------------------------------

class TestAccountEnvIsolation(Fixture):
    def test_each_surface_gets_its_own_isolation_variable(self):
        base = {"PATH": "/usr/bin"}
        claude_env = clgctl.account_env({"kind": "claude", "home": "/tmp/claude-pro1"}, base)
        codex_env = clgctl.account_env({"kind": "codex", "home": "/tmp/codex-cdx1"}, base)
        proxy_env = clgctl.account_env({"kind": "proxy", "dir": "/tmp/ccp-b"}, base)
        self.assertEqual(claude_env["CLAUDE_CONFIG_DIR"], "/tmp/claude-pro1")
        self.assertEqual(codex_env["CODEX_HOME"], "/tmp/codex-cdx1")
        self.assertEqual(proxy_env["CCP_CONFIG_DIR"], "/tmp/ccp-b")

    def test_one_account_env_never_leaks_into_another(self):
        base = {"PATH": "/usr/bin"}
        claude_env = clgctl.account_env({"kind": "claude", "home": "/tmp/claude-pro1"}, base)
        self.assertNotIn("CODEX_HOME", claude_env)
        self.assertNotIn("CCP_CONFIG_DIR", claude_env)
        codex_env = clgctl.account_env({"kind": "codex", "home": "/tmp/codex-cdx1"}, base)
        self.assertNotIn("CLAUDE_CONFIG_DIR", codex_env)

    def test_canonical_claude_identity_file_sits_beside_the_config_dir(self):
        """Verified on 2.1.222: ~/.claude.json for the default account, but
        <dir>/.claude.json once CLAUDE_CONFIG_DIR is set. Getting this backwards
        makes every enrolled account read the canonical account's identity."""
        canonical = clgctl.claude_identity_file(clgctl.CANONICAL_CLAUDE_HOME)
        self.assertEqual(canonical, clgctl.HOME / ".claude.json")
        isolated = clgctl.claude_identity_file(pathlib.Path("/tmp/claude-pro1"))
        self.assertEqual(isolated, pathlib.Path("/tmp/claude-pro1/.claude.json"))

    def test_keychain_service_matches_the_binary_formula(self):
        """`Claude Code-credentials` + `-sha256(config dir)[:8]`, no suffix when
        CLAUDE_CONFIG_DIR is unset. Read out of the 2.1.222 bundle."""
        import hashlib

        self.assertEqual(
            clgctl.keychain_services(clgctl.CANONICAL_CLAUDE_HOME),
            ["Claude Code-credentials"],
        )
        home = pathlib.Path("/tmp/claude-pro1")
        digest = hashlib.sha256(str(home).encode()).hexdigest()[:8]
        self.assertIn(f"Claude Code-credentials-{digest}", clgctl.keychain_services(home))
        # /tmp resolves to /private/tmp on macOS; both spellings must be tried.
        real = hashlib.sha256(os.path.realpath(home).encode()).hexdigest()[:8]
        self.assertIn(f"Claude Code-credentials-{real}", clgctl.keychain_services(home))


# ---------------------------------------------------------------------------
# Routage GPT
# ---------------------------------------------------------------------------

class TestRoutingDefaults(Fixture):
    def test_absent_file_reproduces_todays_hardcoded_routing(self):
        """The externalisation must be a no-op until someone writes the file."""
        self.use_routing(None)
        cfg = clg.load_routing()
        self.assertEqual(cfg["main_model"], "claude-opus-5")
        self.assertEqual(cfg["main_model_env"], "claude-opus-5[1m]")
        self.assertEqual(cfg["tiers"]["opus"], "gpt-5.6-sol")
        self.assertEqual(cfg["tiers"]["sonnet"], "gpt-5.6-terra")
        self.assertEqual(cfg["tiers"]["haiku"], "gpt-5.6-luna")
        self.assertEqual(cfg["tiers"]["inherited"], "gpt-5.6-terra")

    def test_clg_and_clgctl_agree_on_the_defaults(self):
        """Two files, one contract: a drift here silently splits the fleet."""
        self.assertEqual(clg.DEFAULT_ROUTING, clgctl.DEFAULT_ROUTING)

    def test_file_values_override_defaults(self):
        self.use_routing({"tiers": {"haiku": "gpt-5.6-terra"}})
        self.assertEqual(clg.load_routing()["tiers"]["haiku"], "gpt-5.6-terra")
        self.assertEqual(clg.load_routing()["tiers"]["opus"], "gpt-5.6-sol")

    def test_per_account_override_applies_only_to_that_account(self):
        self.use_routing({"overrides": {"b": {"tiers": {"haiku": "gpt-5.6-sol"}}}})
        self.assertEqual(clg.load_routing("b")["tiers"]["haiku"], "gpt-5.6-sol")
        self.assertEqual(clg.load_routing("@b")["tiers"]["haiku"], "gpt-5.6-sol")
        self.assertEqual(clg.load_routing()["tiers"]["haiku"], "gpt-5.6-luna")
        self.assertEqual(clg.load_routing("a")["tiers"]["haiku"], "gpt-5.6-luna")


class TestRoutingNegativeControls(Fixture):
    """Seeded defects. Without these the validator could accept anything and
    every test above would still pass."""

    def test_launcher_falls_back_to_defaults_on_a_corrupt_file(self):
        self.use_routing("{ this is not json")
        cfg = clg.load_routing()
        self.assertEqual(cfg["tiers"]["opus"], "gpt-5.6-sol")

    def test_validator_rejects_an_unknown_tier(self):
        with self.assertRaises(ValueError):
            clgctl.validate_routing({"tiers": {"ultra": "gpt-5.6-sol"}})

    def test_validator_rejects_an_empty_model_name(self):
        with self.assertRaises(ValueError):
            clgctl.validate_routing({"tiers": {"haiku": "   "}})
        with self.assertRaises(ValueError):
            clgctl.validate_routing({"main_model": ""})

    def test_validator_rejects_a_non_object_payload(self):
        with self.assertRaises(ValueError):
            clgctl.validate_routing(["gpt-5.6-sol"])

    def test_validator_rejects_an_unknown_tier_inside_an_override(self):
        with self.assertRaises(ValueError):
            clgctl.validate_routing({"overrides": {"b": {"tiers": {"ultra": "x"}}}})

    def test_validator_accepts_the_shipped_defaults(self):
        """The negative control's control: the validator must not reject everything."""
        self.assertEqual(
            clgctl.validate_routing(clgctl.DEFAULT_ROUTING)["tiers"],
            clgctl.DEFAULT_ROUTING["tiers"],
        )

    def test_dry_run_writes_nothing(self):
        path = self.use_routing(None)
        wrote = clgctl.write_routing(clgctl.DEFAULT_ROUTING, dry_run=True)
        self.assertFalse(wrote)
        self.assertFalse(path.exists())

    def test_apply_writes_and_is_idempotent(self):
        path = self.use_routing(None)
        clgctl.write_routing(clgctl.DEFAULT_ROUTING, dry_run=False)
        self.assertTrue(path.exists())
        first = path.read_text()
        clgctl.write_routing(clgctl.DEFAULT_ROUTING, dry_run=False)
        self.assertEqual(path.read_text(), first)


class TestConfigRoundTrip(Fixture):
    """A config tool that can only add is a one-way ratchet: whatever it writes
    into the founder's live routing can never be taken back out."""

    def config(self, action, assignment=None, account=None, dry_run=False):
        args = types.SimpleNamespace(
            action=action, assignment=assignment, account=account, dry_run=dry_run
        )
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            clgctl.cmd_config(args)
        return buffer.getvalue()

    def test_set_then_unset_returns_to_the_default(self):
        self.use_routing(None)
        self.config("set", "tiers.opus=gpt-5.6-probe")
        self.assertEqual(clgctl.load_routing()["tiers"]["opus"], "gpt-5.6-probe")
        self.config("unset", "tiers.opus")
        self.assertEqual(
            clgctl.load_routing()["tiers"]["opus"],
            clgctl.DEFAULT_ROUTING["tiers"]["opus"],
        )

    def test_account_override_can_be_removed_entirely(self):
        self.use_routing(None)
        self.config("set", "tiers.haiku=gpt-5.6-onlyb", account="b")
        self.assertEqual(clgctl.load_routing("b")["tiers"]["haiku"], "gpt-5.6-onlyb")
        self.config("unset", "tiers.haiku", account="b")
        self.assertEqual(clgctl.load_routing()["overrides"], {})
        self.assertEqual(
            clgctl.load_routing("b")["tiers"]["haiku"],
            clgctl.DEFAULT_ROUTING["tiers"]["haiku"],
        )

    def test_unset_refuses_an_unknown_key(self):
        self.use_routing(None)
        with self.assertRaises(SystemExit):
            self.config("unset", "tiers.ultra")
        with self.assertRaises(SystemExit):
            self.config("unset", "nonsense")

    def test_set_refuses_a_non_tier_key_for_an_account_override(self):
        self.use_routing(None)
        with self.assertRaises(SystemExit):
            self.config("set", "main_model=x", account="b")


# ---------------------------------------------------------------------------
# Fuite de secrets
# ---------------------------------------------------------------------------

class TestNoSecretLeak(Fixture):
    SECRET = re.compile(r"sk-[A-Za-z0-9]|ey[A-Za-z0-9]{10,}|accessToken|refreshToken|id_token")

    def test_status_output_carries_no_token_material(self):
        self.use_registry(MIXED_REGISTRY)
        args = types.SimpleNamespace(on="all", deep=False)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            clgctl.cmd_status(args)
        output = buffer.getvalue()
        self.assertTrue(output.strip(), "status must print something to be auditable")
        self.assertIsNone(self.SECRET.search(output), f"secret-shaped text in status output:\n{output}")

    def test_the_secret_pattern_can_actually_fire(self):
        """Negative control for the guard above."""
        self.assertIsNotNone(self.SECRET.search('{"accessToken": "x"}'))


if __name__ == "__main__":
    unittest.main(verbosity=2)
