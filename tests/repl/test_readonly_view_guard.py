"""Tests for the read-only sub-agent view guard in ``handle_slash_command``.

While the REPL observes a running sub-agent read-only (dived in via the ↓
menu, ``_readonly_view`` set), the plain-message send path is already refused.
Mutating slash-commands must be refused too — running them would unbind or
mutate the *viewed* child, the exact thing read-only view exists to prevent:

* ``/switch`` -> ``switch_to_session()`` unbinds the child + clears the view
* ``/new`` / ``/clear`` -> ``start_new_conversation()`` does the same
* ``/fork`` -> forks + re-points off the child
* ``/model`` / ``/effort`` -> PATCH settings onto the viewed child's session
* ``/compact`` -> posts a compaction turn into the child's conversation
* ``/cancel`` -> cancels the child's in-flight response

Navigational / read-only commands (``/help`` etc.) must still work.

Uses the same stub-and-capture pattern as ``test_repl_fork_command.py``.
"""

from __future__ import annotations

from io import StringIO

import pytest

from omnigent.repl import _repl as repl_mod
from omnigent.repl._repl import _READONLY_BLOCKED_COMMANDS, handle_slash_command

# ``asyncio_mode = "auto"`` (pyproject) collects the async tests below without
# an explicit marker, so the sync ``test_blocked_set_*`` stays unmarked.


# ── Stubs ────────────────────────────────────────────────


class _RecordingSession:
    """Session stub that records every mutating call.

    The guard short-circuits before dispatch, so for a blocked command none
    of these should ever be invoked; ``calls`` staying empty is the assertion.
    """

    def __init__(self, *, readonly: bool, is_streaming: bool = False) -> None:
        self._readonly_view = readonly
        self.model = "test-agent"
        self.session_id: str | None = "conv_child"
        self.is_streaming = is_streaming
        self.calls: list[object] = []

    async def switch_to_session(self, new_session_id: str) -> str:
        self.calls.append(("switch_to_session", new_session_id))
        return new_session_id

    async def start_new_conversation(self) -> None:
        self.calls.append("start_new_conversation")

    def switch_session(self, new_session_id: str) -> None:
        self.calls.append(("switch_session", new_session_id))

    def set_model_override(self, model: str | None) -> None:
        self.calls.append(("set_model_override", model))

    def compact(self) -> None:
        self.calls.append("compact")

    async def cancel(self) -> None:
        self.calls.append("cancel")


class _CapturingHost:
    """Host stub that records ``output()`` calls and renders them to text."""

    def __init__(self) -> None:
        self.outputs: list[object] = []

    def output(self, item: object) -> None:
        self.outputs.append(item)

    def render_plain(self) -> str:
        from rich.console import Console

        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=200, color_system=None)
        for item in self.outputs:
            console.print(item)
        return buf.getvalue()


class _StubFmt:
    muted = "dim"
    accent = "bold"


class _ExplodingSessionsNamespace:
    """Any client.sessions access while blocked is a bug — the guard returns
    before the handler runs, so no client call should happen."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"client.sessions.{name} called while in read-only view")


class _StubClient:
    def __init__(self) -> None:
        self.sessions = _ExplodingSessionsNamespace()


# ── Tests ────────────────────────────────────────────────


def test_blocked_set_covers_every_named_mutating_command() -> None:
    """The denylist must contain every command that unbinds / mutates the
    viewed child. A regression that drops one would silently re-open the hole."""
    assert _READONLY_BLOCKED_COMMANDS.issuperset(
        {
            "/switch",
            "/new",
            "/clear",
            "/fork",
            "/model",
            "/effort",
            "/compact",
            "/cancel",
        }
    )
    # Every blocked command is a real registered command (no typos).
    for cmd in _READONLY_BLOCKED_COMMANDS:
        assert cmd in repl_mod.COMMANDS, f"{cmd} is blocked but not a registered command"


@pytest.mark.parametrize("cmd", sorted(_READONLY_BLOCKED_COMMANDS))
async def test_mutating_command_blocked_in_readonly_view(cmd: str) -> None:
    """In read-only view, each mutating command is refused: the session is
    never touched and the user is told to press ← first."""
    session = _RecordingSession(readonly=True)
    host = _CapturingHost()
    client = _StubClient()

    # Pass an arg so commands that branch on one (``/switch <id>``,
    # ``/model x``) would take their mutating path if the guard let them.
    await handle_slash_command(f"{cmd} arg", session, client, host, _StubFmt())

    assert session.calls == [], (
        f"{cmd} mutated/unbound the viewed child in read-only view: {session.calls}"
    )
    out = host.render_plain().lower()
    assert "read-only view" in out
    assert "←" in host.render_plain()


async def test_mutating_command_runs_when_not_in_readonly_view() -> None:
    """Positive control: the guard is conditional. With ``_readonly_view``
    false the same command dispatches normally to its handler."""
    # ``/compact`` while streaming just prints a hint and returns — no
    # mutation — so reaching that message proves dispatch was NOT blocked.
    session = _RecordingSession(readonly=False, is_streaming=True)
    host = _CapturingHost()

    await handle_slash_command("/compact", session, _StubClient(), host, _StubFmt())

    out = host.render_plain().lower()
    assert "read-only view" not in out
    assert "cannot compact while a response is running" in out


async def test_navigational_command_allowed_in_readonly_view() -> None:
    """Read-only / navigational commands (``/help``) stay available while
    observing a sub-agent — they don't touch the child."""
    session = _RecordingSession(readonly=True)
    host = _CapturingHost()

    await handle_slash_command("/help", session, _StubClient(), host, _StubFmt())

    out = host.render_plain()
    assert "read-only view" not in out.lower()
    # /help lists registered commands.
    assert "/switch" in out
    assert session.calls == []
