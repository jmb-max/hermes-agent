# claude-worker

Delegate coding tasks to an isolated Claude Code CLI running inside a pinned,
locked-down Docker sandbox, authenticated with the installed Claude Code OAuth
session (never an API key).

Opt-in: the plugin is dark on disk until it is listed in `plugins.enabled`.

## Scope: every Discord-origin session

Scope is the **platform**, not a channel list.

* **In scope** — every Discord-origin session: any guild channel, any DM, and
  **every thread** under either. A thread needs no parent-channel lookup,
  because the platform (not the chat id) is the test — which is what closed
  the old hole where a thread's `chat_id` is the *thread* id and a
  channel-list check silently missed it.
* **Out of scope** — every other platform, and the CLI (no chat identity at
  all).
* **Unconfirmable** — a session that is plainly a real chat (it has a chat id
  and/or a thread id) whose platform cannot be confirmed from the dispatch
  cache, the session contextvars, or the session key. This is `UNKNOWN`, and
  the gate **fails closed** on it: a gated mutation inside a Git worktree is
  blocked with an actionable message rather than waved through as
  "probably not Discord". So is any error reading identity, the cache, or
  config.

Provenance comes from the **explicit session key** the dispatcher passes with
every `pre_tool_call`, as well as the ambient session ContextVars and the
recorded dispatch identity. The explicit key is what makes this survive a
context compression / continuation: `clear_session_vars` resets every ambient
var to `""` (which deliberately suppresses the `os.environ` fallback), so a
live Discord session can arrive with no ambient chat id, platform, or session
key at all. Reading that as "no chat identity → this must be the CLI" answered
a confident *out of scope* and dropped both the write gate and the terminal
guard mid-session. The explicit key is now consulted first — as the identity
cache lookup key, and for the platform segment it carries directly.

There is no channel allowlist. Configuration cannot widen scope, and it can no
longer narrow it either — the only knob is a global kill switch
(`discord.enabled: false`, or the deprecated `canary.enabled: false`), which is
safe precisely because it removes the restriction from *every* session at once
instead of quietly exempting one.

## What is gated

In a Discord-origin session:

1. **Direct repo mutations** — `patch`, `write_file`, and every mutating
   `skill_manage` action are blocked when the target resolves inside a Git
   worktree, per `(session, resolved Git root)`, **for the whole session**.
   The worker is the coder: every code change goes through `claude_worker`.

   A **successful `claude_worker` run releases nothing.** It used to release
   that `(session, repo)` pair, which made the worker a one-time toll gate —
   run it once, then edit the repository directly for the rest of the session
   — and that is what was observed in production: one successful Opus run,
   then a stream of direct `patch`/`write_file` calls, then a host `claude -p`
   through `terminal`. A successful run now proves the worker works and grants
   nothing.

   The **one** release is an explicit Terra fallback that actually returned a
   result (full provenance required: `fallback_requested`,
   `fallback_provenance="terra_auxiliary"`, and a nested
   `fallback.ready`/`fallback.ok` with non-empty notes). It exists because a
   session whose worker cannot run *at all* would otherwise have no way
   forward; a session whose worker can run has one already.

   An **open circuit breaker does not release anything** — it HOLDs. Releases
   are in-process only and reset on restart, which is the safe direction.

2. **Host Claude CLI invocations through `terminal`** — refused outright,
   unlock state or not, because that path sidesteps the entire sandbox (image
   pin, non-root uid, empty MCP config, tool allowlist, resource caps). This
   covers the bare command `claude`, any path ending in `/claude`, `claude -p`,
   wrapper forms (`env`, `env FOO=1`, `env -i`, `sudo -n`, `command`, `nohup`,
   `timeout 300`, `exec`, `nice`, …), executors that run a command from their
   argument list (`xargs claude -p x`, `watch claude`, `parallel claude … :::`,
   `find . -exec claude -p x \;`, `find . -execdir …`), GNU `env`'s
   split-string form (`env -S "claude -p x"`, `env --split-string=…`), busybox
   applet dispatch (`busybox sh -c "claude -p x"`, `busybox env claude`), any
   segment of a shell-chained command (`make && claude -p x`, `x; claude`,
   `a | claude`, `printf x | xargs claude -p`), payloads smuggled through an
   explicit shell (`bash -c "claude -p x"` — including behind an option
   cluster, `bash -cx`/`sh -cv`/`bash -ce`, where the command string is the
   next word, and behind options with values of their own, `bash -O extglob
   -c …`, `bash --rcfile /dev/null -c …`, `bash --noprofile --norc -c …`),
   a script fed to a shell as visible text (`bash <<< "claude -p x"`, and
   here-doc bodies, which are ordinary command lines), segments headed by
   shell grammar rather than a command (`if true; then claude -p x; fi`,
   `while true; do claude -p x; done`, `! claude -p x`, `{ claude -p x; }`,
   `f(){ claude -p x; }; f`, `case x in x) claude -p x;; esac`), and
   executable substitutions, which the shell runs as commands of their own
   (`` echo `claude -p x` ``, `echo $(claude -p x)`, `cat <(claude -p x)`,
   `cat >(claude -p x)`). Shell payloads and substitutions are followed
   recursively to a bounded depth. A command whose quoting does not tokenize
   is blocked only when a direct invocation is still visible in a raw scan.

   Ordinary terminal work is untouched: `git commit -m 'claude worker fix'`,
   `grep -r claude .`, `echo claude`, `printf 'claude'`, `find . -name claude`,
   `ls /usr/local/bin/claude`, `pytest tests/test_claude_worker_gate.py`,
   `python3 -c 'print("claude")'`, `make build && pytest -q`, `bash -c 'echo
   claude'`, `bash -O extglob -c 'echo claude'`, `echo "then claude"`,
   `printf 'do claude'`, `git commit -m 'if claude'` and
   `echo 'bash <<< "claude -p x"'` all mention or resemble claude and none of
   them runs it: a reserved word only counts in command-head position, and a
   quoted mention is an argument. Single quotes suppress substitution, so
   `echo '$(claude)'` prints a string and stays allowed.

   **What this does not cover.** It is a lexical guard over one command
   string, not a sandbox. It blocks direct invocations and the common
   shell-native wrappers, executors and substitutions above; it cannot see a
   command that builds its target at runtime — an interpreter decoding a
   string and calling `exec`, a variable the shell expands to the binary's
   name, a script file that runs claude itself. Those contain no lexical
   `claude` for any parser to find, and widening the guard to guess at them
   would block ordinary work without closing the hole. Two neighbours of the
   stdin rule are named rather than implied: a shell reading a script from a
   file (`bash script.sh`, `bash < script.sh`) has no command text in the
   string, and a script arriving through a pipe (`echo "claude -p x" | bash`)
   is not analysed either. Containment for those is the sandbox and the
   credential boundary.

Everything else proceeds: another tool, a non-mutating `skill_manage` action, a
path outside every Git worktree, a non-Discord session.

**No deadlocks outside Git.** A path outside every Git worktree is never gated,
even in a confirmed Discord session — the worker could not run there either, so
gating it would leave nothing able to satisfy the gate.

## Project scope: dynamic, per request, one repository

There is no `gate.repo_roots` authority anymore. `project.py` is the single
resolver that both the gate and the runner call, so the two can never disagree
and manufacture a deadlock (the gate can never block a repository the worker is
not allowed to run in).

Given a requested path it returns the **canonical Git worktree root** that
contains it, and that exact root — never a parent holding several checkouts —
is what gets bind-mounted at `/workspace`.

Accepted: an existing, absolute path whose canonical form lives inside a real
Git worktree. Rejected, every one a hard refusal:

* a relative path, an empty path, or one containing a NUL byte;
* a path that does not exist, or is not a directory;
* a directory inside no Git worktree at all, and a **bare** repository;
* a **symlink escape** — `realpath` runs first, so a link planted inside a repo
  that points outside it is judged where it really goes;
* a `.git` entry that is a symlink or is neither a directory nor a regular
  file, and a `.git` **file** whose contents are not the `gitdir:` pointer git
  itself writes for a linked worktree or submodule (linked worktrees and
  submodules are supported);
* a resolved root that is system-sensitive (`/`, `/etc`, `/usr`, `/var`, …), a
  whole home directory, or the multi-project container `/root/worktrees`.

Sibling prefixes are not containment: `/repo-evil` is never "inside" `/repo` —
every containment test is segment-wise.

The filesystem ancestor walk is the primary resolver (deterministic, no
subprocess, works for `.git` files). A trusted git binary is consulted as a
second opinion, and a **disagreement is fatal** rather than ignored. That
subprocess is hardened on every axis: the absolute `policy.GIT_BIN` only, whose
entire path chain must be root-owned, non-symlink and never group/world-
writable *before* it runs; a fixed argv list, never a shell; a bounded timeout;
an environment built from scratch with `GIT_CONFIG_NOSYSTEM` /
`GIT_CONFIG_GLOBAL=/dev/null` / `GIT_CONFIG_SYSTEM=/dev/null`; explicit `-c`
overrides (`core.hooksPath=/dev/null`, `core.fsmonitor=false`,
`protocol.ext.allow=never`); `safe.directory=*` as the literal glob rather than
the requested path interpolated into a config value; and strict output
validation.

## OAuth freshness preflight

Before any spawn — after cwd validation and after the already-open-breaker
HOLD check, and *before* baseline evidence gathering or any docker/Claude
container — `oauth.py` reads the freshness of the installed Claude Code OAuth
credential.

This exists because the breaker is the right response to an **observed**
authentication failure and the wrong response to a credential we could have
inspected first: an already-expired token used to come back as a 401, classify
as `auth`, and hold every session for a full hour over something a refresh
would have fixed in a second.

* **Fresh** — proceed normally. Nothing about the run changes.
* **Anything else** (missing, malformed or oversized, no OAuth material,
  expired, expired-but-refreshable with no probe installed) — a **HOLD**:
  `status: "HOLD"`, `attempts: 0`, `failure_class: "auth_preflight"`, no
  container started, and the session stays gated. The result carries the
  state name (`oauth_state`) and `oauth_refresh_attempted`, and nothing else
  from the credential — never a token, never an expiry, never the free-text
  reason a probe produced.
* **No breaker mutation, in either direction.** The preflight never opens a
  class (an unreadable file is not an API rejection), and never clears or
  resets one either — a breaker opened by a real failure keeps its own
  cooldown. `auth_preflight` is deliberately not one of the breaker classes.
  Observed 401s during a real run still open the `auth` breaker exactly as
  before.

**Refresh is not reimplemented here — it is delegated.** Writing a refreshed
token back to the root-owned credentials path would mean owning precisely the
privileged credential-writing logic this plugin never owns, and `oauth.py`
executes nothing at all. A refreshable-expired credential instead gets **at
most one** isolated probe through the `oauth.REFRESH_PROBE` injection point,
followed by a fresh re-read from disk to see whether the credential actually
recovered — the preflight believes the file, never the probe's return value.

`register()` installs exactly one probe: `oauth_refresh.refresh_probe`, which
hands the refresh to the program that legitimately owns the privileged write
and knows the grant parameters — **the host's own Claude CLI** at the fixed
absolute `/usr/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe` (the
packaged executable itself, not the `claude` symlink npm puts on `$PATH`, which
trust validation refuses). Everything about that run is a literal in
`oauth_refresh.py` and unreachable from config: the path chain is
trust-validated (root-owned, non-symlink, owner-executable) immediately before
each run, and the identity snapshot that validation returns is **re-checked
immediately before process creation**, so a binary swapped while the probe
waited on the lock fails closed instead of being executed; the argv is fixed
(`--print`, one turn, `--disallowed-tools "*"` plus a `--settings` payload
whose `permissions.deny` is `["*"]` with no allow/ask list, `--permission-mode
dontAsk` so the turn neither prompts nor bypasses those rules, an empty
`--strict-mcp-config`, no setting sources, no session/resume, the cheap model,
one fixed prompt — note that an empty *allow* list would deny nothing, since
an allow list only ever adds a permission); the environment is built from
scratch (`HOME`, `PATH`, `CLAUDE_CONFIG_DIR`, `LANG` only, so no ambient API
key, base URL, proxy or loader variable can redirect or hijack it); the cwd is
an empty throwaway directory created inside the validated root-owned mode-0700
`policy.CREDENTIAL_STAGING_ROOT` — never a repository, and never the ambient
temp root a hostile `TMPDIR` could move; the run is bounded by a timeout, and a
timed-out turn is reaped by **process group** (SIGTERM, a short fixed grace,
then SIGKILL, all before the cwd is removed) so the CLI's own descendants are
never orphaned; and its stdout/stderr go to `/dev/null` so a diagnostic quoting
the credential never enters the process. Concurrent sessions serialize on an
isolated root-owned mode-0600 lock under `policy.CREDENTIAL_STAGING_ROOT`, and
freshness is re-read **inside** the lock, so the loser of the race spawns
nothing. A clean exit is not a refresh: the credential is re-read while the
lock is still held, and only a credential that is now fresh is reported as one.
Every failure mode (knob off, untrusted binary, a binary that changed after
validation, an unsafe staging directory, busy lock, timeout, non-zero exit,
unreadable status, a run that renewed nothing, unexpected error) returns one of
a fixed, non-secret reason literals and the preflight HOLDs.

Operator knobs (`plugins.entries.claude-worker.oauth`): `auto_refresh`
(default `true`; only a literal `false` turns it off, and then no probe is
installed at all) and `refresh_timeout_seconds` (default 45, clamped to
10–120). Nothing else in that section survives the loader.

**Repeated `refreshable_expired` HOLDs.** If they persist, the probe is not
running or not recovering the credential — most often because the host CLI at
`/usr/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe` is absent, is a
symlink, or is not root-owned, which
fails closed rather than executing an unvetted binary. Note also what is still
*not* done: refreshing into the per-spawn credential staging directory instead
of the host file was evaluated and rejected — an OAuth refresh grant may
rotate the refresh token, so a deliberately non-persisted refresh can spend the
host's only refresh token and turn a recoverable `refreshable_expired` into an
unrecoverable `expired` that needs a full interactive re-login, for every
consumer of that credential. The grant parameters also are not in the
credential file, and `oauth.py` is tripwired against the network and write
capabilities it would need. See the `oauth.py` module docstring for the full
reasoning. The always-available fix is to **re-authenticate the host Claude
Code session** (what the HOLD message tells you to do).

Secrets never leave the module: the credential bytes are read into a local,
parsed, and dropped, only non-secret metadata is returned, and any free text
that reaches a result or a log line is scrubbed of the values currently in the
credentials file (including a probe's own exception message, which is logged
by type plus redacted text rather than as a traceback).

## Models and routing

`claude-sonnet-5` is the default for every task. `claude-opus-5` is used for an
architecture / security / hard-debugging classification, or for the single
controlled escalation after one non-breaker Sonnet failure. Both are policy
literals, unreachable from configuration, and the attempt cap is structurally
2. Hermes' own Sol/Terra/Grok routing is a separate system and is untouched.

## Configuration

Everything lives under `plugins.entries.claude-worker`. This is a *validating*
loader: only known keys with known types survive, anything else is dropped, and
nothing policy-critical is in this namespace at all.

```yaml
plugins:
  entries:
    claude-worker:
      discord:
        enabled: true          # global kill switch; the ONLY scope knob
      isolation:
        timeout_seconds: 900   # clamped to [30, 3600]
      breaker:
        cooldown_seconds:
          auth: 3600
          rate: 900
          extra_usage: 3600
      review:
        enabled: true
        min_changed_files: 3
      verification:
        enabled: false
        command: []            # fixed operator argv, run in its own
                               # network-isolated, credential-less container
```

### Deprecated, still parsed, inert

| Key | Status |
| --- | --- |
| `canary.enabled` | Deprecated **alias** for `discord.enabled`; still honored. |
| `canary.channel_ids` | Deprecated and inert — scope is the platform, so there is no channel list to narrow. |
| `gate.repo_roots` | Deprecated and inert — project scope is resolved dynamically per request. |
| `telemetry.path` | Deprecated and inert — telemetry always writes to `$HERMES_HOME/claude-worker/telemetry.jsonl`. |

A stale config carrying any of these keeps loading; none of them can change
behavior. `gate.repo_roots` in particular is inert *because* it used to be
authoritative: whatever it named was what got mounted, so a well-meaning entry
of `/root/worktrees` handed every sandbox container every sibling project at
once.

## Tests

```
pytest tests/plugins/test_claude_worker_canary.py \
       tests/plugins/test_claude_worker_gate.py \
       tests/plugins/test_claude_worker_project.py \
       tests/plugins/test_claude_worker_terminal_guard.py \
       tests/plugins/test_claude_worker_policy.py \
       tests/plugins/test_claude_worker_config.py \
       tests/plugins/test_claude_worker_plugin_registration.py \
       tests/plugins/test_claude_worker_runner.py \
       tests/plugins/test_claude_worker_oauth.py \
       tests/plugins/test_claude_worker_isolation.py
```

The gate and terminal-bypass tests assert through the REAL enforcement path
(`hermes_cli.plugins.resolve_pre_tool_block`), so they prove the actual
tool-dispatch sites block rather than that a callback returned a dict.
