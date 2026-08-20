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
import sys
import types
from pathlib import Path

_NS_PARENT = "hermes_plugins"
_SLUG = "claude_worker_under_test"


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
