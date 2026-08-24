"""Shared loader for the claude-worker plugin package under test.

``plugins/claude-worker`` has a hyphen in its directory name, so it cannot be
imported with a normal ``import plugins.claude_worker`` statement. The real
plugin loader (``hermes_cli/plugins.py``) works around this by exec'ing the
plugin's ``__init__.py`` as a synthetic package ``hermes_plugins.<slug>`` with
``submodule_search_locations`` pointed at the plugin directory, so the
package's own relative imports (``from . import config``) resolve normally.
Tests mirror that exact mechanism (see ``tests/plugins/test_security_guidance_plugin.py``)
so submodule-level unit tests exercise the same import shape as production.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

_NS_PARENT = "hermes_plugins"
_SLUG = "claude_worker_under_test"

#: Prefixes ``plugins/claude-worker/policy.py`` refuses to treat as a project
#: root at any depth. ``tmp_path`` normally lives under ``/tmp`` (which is
#: fine — only ``/tmp`` ITSELF is unsafe, not paths beneath it), but a
#: ``TMPDIR`` pointing into ``/var`` or ``/run`` would make every dynamic-root
#: test vacuous, so those runs skip loudly instead of quietly passing.
_UNSAFE_TMP_PREFIXES = (
    "/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64", "/libx32",
    "/proc", "/run", "/sbin", "/sys", "/usr", "/var",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def plugin_dir() -> Path:
    return repo_root() / "plugins" / "claude-worker"


def _ensure_namespace_parent() -> None:
    if _NS_PARENT not in sys.modules:
        ns_pkg = types.ModuleType(_NS_PARENT)
        ns_pkg.__path__ = []
        ns_pkg.__package__ = _NS_PARENT
        sys.modules[_NS_PARENT] = ns_pkg


def _ensure_lightweight_package(force: bool = False) -> str:
    """Register the plugin dir as an importable package WITHOUT executing
    its real ``__init__.py``.

    This lets submodule-level unit tests (``routing.py``, ``breaker.py``,
    ...) import a single sibling module before every other sibling exists —
    the RED phase of TDD would otherwise be blocked on the whole package
    (since a real ``__init__.py`` typically does ``from . import <every
    submodule>``).
    """
    module_name = f"{_NS_PARENT}.{_SLUG}"
    _ensure_namespace_parent()
    if force or module_name not in sys.modules:
        pdir = plugin_dir()
        pkg = types.ModuleType(module_name)
        pkg.__path__ = [str(pdir)]
        pkg.__package__ = module_name
        sys.modules[module_name] = pkg
    return module_name


def load_submodule(name: str, force: bool = False) -> types.ModuleType:
    """Import one submodule (e.g. ``"canary"``) of the plugin package.

    Uses the lightweight (non-executing) parent package, so this works even
    when sibling submodules don't exist yet — exactly what RED-phase tests
    need. Re-imports fresh when ``force=True`` (useful when a test wants a
    clean module-level cache, e.g. the preflight-probe cache in ``runner``).
    """
    module_name = _ensure_lightweight_package(force=force)
    full_name = f"{module_name}.{name}"
    if force and full_name in sys.modules:
        del sys.modules[full_name]
    return importlib.import_module(full_name)


def require_safe_tmp(tmp_path: Path) -> None:
    """Skip when ``tmp_path`` itself sits inside a system-sensitive tree.

    The dynamic project-root resolver refuses such roots outright, so a repo
    built there could never be resolved and a "this is gated" assertion would
    pass for entirely the wrong reason. Fail visibly (skip) rather than
    silently.
    """
    import pytest

    real = os.path.realpath(str(tmp_path))
    for prefix in _UNSAFE_TMP_PREFIXES:
        if real == prefix or real.startswith(prefix.rstrip("/") + "/"):
            pytest.skip(f"tmp_path {real!r} is inside the system-sensitive tree {prefix!r}")


def stub_fresh_oauth_preflight(monkeypatch) -> list:
    """Make ``oauth.preflight`` report a healthy, non-expired credential.

    ``runner.run_worker`` runs the OAuth freshness preflight before it spawns
    anything, and production reads the fixed root-owned
    ``policy.HOST_CREDENTIALS_PATH``, which does not exist under a test
    ``tmp_path`` home. Without this stub every orchestration test would
    exercise the auth-HOLD path instead of the behavior it is actually about.

    Returns the list the stub appends one entry to per call, so a caller can
    assert the preflight ran (and ran exactly once). Tests that are ABOUT the
    preflight override it again with their own stub, or point
    ``policy.HOST_CREDENTIALS_PATH`` at a real temporary credentials file and
    let the production implementation run for real.
    """
    oauth = load_submodule("oauth")
    calls: list = []

    def _fresh(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "ok": True,
            "state": oauth.STATE_FRESH,
            "classification": None,
            "refresh_attempted": False,
            "reason": "",
            "freshness": {"state": oauth.STATE_FRESH, "expired": False},
        }

    monkeypatch.setattr(oauth, "preflight", _fresh)
    return calls


def write_credentials(
    path: Path,
    *,
    access_token: str = "test-access-token-value",
    refresh_token: str | None = "test-refresh-token-value",
    expires_at: float | int | None = None,
    section: str = "claudeAiOauth",
) -> Path:
    """Write a Claude-Code-shaped OAuth credentials file at *path*.

    ``expires_at`` is written verbatim, so a caller controls whether it is
    seconds or milliseconds since the epoch (``oauth._normalize_expiry``
    accepts both). ``refresh_token=None`` omits the refresh key entirely.
    """
    body: dict = {"accessToken": access_token}
    if refresh_token is not None:
        body["refreshToken"] = refresh_token
    if expires_at is not None:
        body["expiresAt"] = expires_at
    body["scopes"] = ["user:inference", "user:profile"]
    body["subscriptionType"] = "max"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({section: body}), encoding="utf-8")
    return path


def make_git_repo(base: Path, name: str = "repo", *, marker: str = "dir") -> Path:
    """Create a directory the dynamic resolver accepts as a Git worktree root.

    *marker* ``"dir"`` writes an ordinary ``.git`` DIRECTORY (a normal
    checkout); ``"file"`` writes a ``.git`` FILE holding the ``gitdir:``
    pointer git itself writes for a linked worktree or a submodule. Both are
    valid worktree markers, and covering both here is the point: the ``.git``
    file shape is exactly what a static allowlist and a naive ``isdir(".git")``
    check used to miss.

    No ``git init`` is run, so the real git binary — if it is even trusted on
    this host — has no usable opinion about these directories and the
    filesystem walk stands alone. That keeps every caller deterministic
    regardless of whether ``/usr/bin/git`` passes the trust check in the
    environment the suite happens to run in.
    """
    path = base / name if name else base
    path.mkdir(parents=True, exist_ok=True)
    if marker == "dir":
        (path / ".git").mkdir(exist_ok=True)
    elif marker == "file":
        (path / ".git").write_text(
            f"gitdir: {base / (name + '.gitdir')}\n", encoding="utf-8",
        )
    else:  # pragma: no cover - programming error in a test
        raise ValueError(f"unknown .git marker kind: {marker!r}")
    return path


def load_plugin_package(force: bool = False) -> types.ModuleType:
    """Load the claude-worker plugin package by executing its REAL
    ``__init__.py`` (register(ctx) and all).

    Uses the SAME synthetic package name as :func:`load_submodule` — this is
    deliberate: ``__init__.py``'s ``from . import canary, gate, ...`` must
    resolve to the identical module objects a test may have already loaded
    (and monkeypatched) via ``load_submodule``, and vice versa. Using a
    different package name here would give the real plugin its own private
    copy of e.g. ``canary._eligibility_cache``, silently decoupling
    end-to-end tests from the state a test set up.
    """
    module_name = _ensure_lightweight_package(force=False)
    if not force and getattr(sys.modules.get(module_name), "register", None) is not None:
        return sys.modules[module_name]

    pdir = plugin_dir()
    spec = importlib.util.spec_from_file_location(
        module_name,
        pdir / "__init__.py",
        submodule_search_locations=[str(pdir)],
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = module_name
    module.__path__ = [str(pdir)]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
