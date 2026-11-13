"""Environment settings: ``CODEASPOLICY_*`` variables, read by the package."""

from __future__ import annotations

from codeaspolicy._env import setting, setting_var
from codeaspolicy.generate import provider_overrides


def test_setting_reads_the_codeaspolicy_variable_else_the_default(monkeypatch):
    monkeypatch.delenv("CODEASPOLICY_PRIMITIVES", raising=False)
    assert setting("PRIMITIVES", "full") == "full"
    assert setting_var("PRIMITIVES") == "CODEASPOLICY_PRIMITIVES"
    monkeypatch.setenv("CODEASPOLICY_PRIMITIVES", "no-vla")
    assert setting("PRIMITIVES", "full") == "no-vla"
    monkeypatch.setenv("CODEASPOLICY_PRIMITIVES", "")  # set but empty still counts as set
    assert setting("PRIMITIVES", "no-vla") == ""


def test_codex_retries_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("CODEASPOLICY_CODEX_RETRIES", "7")
    joined = " ".join(provider_overrides("https://gw.example.com", "KEY"))
    assert "request_max_retries=7" in joined and "stream_max_retries=7" in joined
