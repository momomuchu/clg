#!/usr/bin/env python3
"""Behavior locks for the flags clg hands to Claude Code.

The local-only mode (`clgl`) is invisible at runtime: a session that silently
kept ~/.claude looks exactly like one that dropped it, and the difference only
shows up as a kernel the founder asked to leave out. These tests pin the two
things that make the mode real — the source list excludes `user`, and the one
setting clg's own --dangerously-skip-permissions depends on is handed back.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest


REPO = pathlib.Path(__file__).parents[1]
CLG_PATH = REPO / "bin" / "clg"
LOADER = importlib.machinery.SourceFileLoader("clg_launcher_flags", str(CLG_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
clg = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = clg
LOADER.exec_module(clg)


class LaunchFlagsTest(unittest.TestCase):
    def test_default_session_touches_no_setting_source(self):
        flags = clg.launch_flags([], local_only=False)
        self.assertNotIn("--setting-sources", flags)
        self.assertNotIn("--settings", flags)
        self.assertEqual(flags[0], "--remote-control")
        self.assertIn("--dangerously-skip-permissions", flags)

    def test_print_mode_drops_remote_control(self):
        for arg in ("-p", "--print"):
            with self.subTest(arg=arg):
                flags = clg.launch_flags([arg, "hello"], local_only=False)
                self.assertNotIn("--remote-control", flags)
                self.assertIn("--teammate-mode", flags)

    def test_local_only_excludes_the_user_source(self):
        flags = clg.launch_flags([], local_only=True)
        sources = flags[flags.index("--setting-sources") + 1]
        self.assertEqual(sources, "project,local")
        self.assertNotIn("user", sources.split(","))

    def test_local_only_hands_back_the_bypass_prompt_skip(self):
        # clg always passes --dangerously-skip-permissions; its confirmation
        # dialog is suppressed by a user setting the mode just dropped.
        flags = clg.launch_flags([], local_only=True)
        self.assertIn("--dangerously-skip-permissions", flags)
        payload = json.loads(flags[flags.index("--settings") + 1])
        self.assertEqual(payload, {"skipDangerousModePermissionPrompt": True})

    def test_local_only_still_honours_print_mode(self):
        flags = clg.launch_flags(["-p", "x"], local_only=True)
        self.assertNotIn("--remote-control", flags)
        self.assertIn("--setting-sources", flags)

    def test_user_flags_would_land_after_ours(self):
        # main() execs [*launch_flags, *args]; a --setting-sources the user
        # types himself must therefore be the last occurrence and win.
        flags = clg.launch_flags([], local_only=True)
        argv = [*flags, "--setting-sources", "user,project,local"]
        last = len(argv) - 1 - argv[::-1].index("--setting-sources")
        self.assertEqual(argv[last + 1], "user,project,local")


class ClglWrapperTest(unittest.TestCase):
    def run_wrapper(self, *args):
        """Run bin/clgl against a stub `clg` that echoes what it received."""
        with tempfile.TemporaryDirectory() as tmp:
            box = pathlib.Path(tmp)
            shutil.copy(REPO / "bin" / "clgl", box / "clgl")
            stub = box / "clg"
            stub.write_text(
                '#!/bin/sh\n'
                'echo "CLG_LOCAL_ONLY=${CLG_LOCAL_ONLY-unset}"\n'
                'for a in "$@"; do echo "ARG=$a"; done\n'
            )
            for path in (box / "clgl", stub):
                path.chmod(path.stat().st_mode | stat.S_IEXEC)
            out = subprocess.run(
                [str(box / "clgl"), *args],
                capture_output=True, text=True, check=True,
                env={**os.environ, "CLG_LOCAL_ONLY": ""},
            )
            return out.stdout.splitlines()

    def test_wrapper_sets_the_local_flag(self):
        self.assertIn("CLG_LOCAL_ONLY=1", self.run_wrapper())

    def test_wrapper_forwards_args_untouched(self):
        lines = self.run_wrapper("@b", "-p", "salut")
        self.assertEqual(
            [l for l in lines if l.startswith("ARG=")],
            ["ARG=@b", "ARG=-p", "ARG=salut"],
        )

    def test_installer_symlinks_clgl(self):
        tools = (REPO / "install.sh").read_text()
        self.assertRegex(tools, r"for tool in .*\bclgl\b")


if __name__ == "__main__":
    unittest.main()
