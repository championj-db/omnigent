"""Tests for the read-only sub-agent view guard in ``handle_slash_command``.

While the REPL observes a running sub-agent read-only (dived in via the ↓
menu, ``_readonly_view`` set), the plain-message send path is already refused.
Slash-commands must be refused too — unless they're on a small allowlist of
navigational / read-only commands — because anything else either unbinds /
mutates / cancels the *viewed* child or posts a turn into it:

* ``/switch`` -> ``switch_to_session()`` unbinds the child + clears the view
* ``/new`` / ``/clear`` -> ``start_new_conversation()`` does the same
* ``/fork`` -> forks + re-points off the child
* ``/model`` / ``/effort`` -> PATCH settings onto the viewed child's session
* ``/compact`` -> posts a compaction turn into the child's conversation
* ``/cancel`` -> cancels the child's in-flight response
* skill slash-commands (registered at runtime) -> ``send_skill_slash_command``
  posts a turn into the session

The guard is an ALLOWLIST so it stays complete as commands are added (and so it
covers the dynamically-registered skill commands a static denylist couldn't
name). Uses the same stub-and-capture pattern as ``test_repl_fork_command.py``.
"""

from __future__ import annotations

from io import StringIO

import pytest

from omnigent.repl._repl import (
    _READONLY_ALLOWED_COMMANDS,
    COMMANDS,
    handle_slash_command,
    register_skill_commands,
    unregister_skill_commands,
)
from omnigent.spec.types import SkillSpec

# ``asyncio_mode = "auto"`` (pyproject) collects the async tests below without
# an explicit marker, so the sync ``test_*`` helpers stay unmarked.

# Built-in commands that mutate / unbind / cancel the viewed child's session.
# Kept explicit here (rather than imported) so the test pins the *intended*
# behaviour independently of the production allowlist.
_KNOWN_MUTATING_COMMANDS = (
    "/switch",
    "/new",
    "/clear",
    "/fork",
    "/model",
    "/effort",
    "/compact",
    "/cancel",
)


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

    def send_skill_slash_command(self, name: str, arg: str) -> object:
        self.calls.append(("send_skill_slash_command", name, arg))

        async def _gen():  # pragma: no cover - should never run while blocked
            if False:
                yield None

        return _gen()


class _CapturingHost:
    """Host stub that records ``output()`` calls and renders them to text."""

    def __init__(self) -> None:
        self.outputs: list[object] = []

    def output(self, item: object) -> None:
        self.outputs.append(item)

    def start_timer(self) -> None:  # used by the skill handler
        self.outputs.append("<start_timer>")

    def render_plain(self) -> str:
        from rich.console import Console

        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=200, color_system=None)
        for item in self.outputs:
            if isinstance(item, str):
                console.print(item)
            else:
                console.print(item)
        return buf.getvalue()


class _StubFmt:
    muted = "dim"
    accent = "bold"

    def user_message(self, text: str) -> str:
        return f"<user:{text}>"


class _ExplodingSessionsNamespace:
    """Any client.sessions access while blocked is a bug — the guard returns
    before the handler runs, so no client call should happen."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"client.sessions.{name} called while in read-only view")


class _StubClient:
    def __init__(self) -> None:
        self.sessions = _ExplodingSessionsNamespace()


# ── Tests ────────────────────────────────────────────────


def test_allowlist_is_all_real_navigational_commands() -> None:
    """Every allowlisted name is a registered command (no typos), and the
    allowlist contains none of the known mutating commands."""
    for cmd in _READONLY_ALLOWED_COMMANDS:
        assert cmd in COMMANDS, f"{cmd} is allowlisted but not a registered command"
    assert not (_READONLY_ALLOWED_COMMANDS & set(_KNOWN_MUTATING_COMMANDS)), (
        "a mutating command leaked onto the read-only allowlist"
    )


def test_every_registered_command_is_allowlisted_or_blocked() -> None:
    """Completeness: in read-only view, every registered command is either on
    the allowlist or refused. Proves no built-in mutator slips through and
    documents the closure — a denylist couldn't make this guarantee."""
    assert _READONLY_ALLOWED_COMMANDS.issubset(set(COMMANDS)), "stale allowlist entry"
    # The known mutators must all be OFF the allowlist (i.e. blocked).
    for cmd in _KNOWN_MUTATING_COMMANDS:
        assert cmd not in _READONLY_ALLOWED_COMMANDS, f"{cmd} must be blocked in read-only view"


@pytest.mark.parametrize("cmd", _KNOWN_MUTATING_COMMANDS)
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
    out = host.render_plain()
    assert "read-only view" in out.lower()
    assert "←" in out


async def test_skill_command_blocked_in_readonly_view() -> None:
    """A dynamically-registered skill slash-command posts a turn into the
    viewed child via ``send_skill_slash_command`` — it must be refused in
    read-only view. This is the hole a static denylist could not have covered."""
    registered = register_skill_commands(
        [SkillSpec(name="code-review", description="Review the diff", content="")]
    )
    try:
        assert "/code-review" in COMMANDS  # sanity: it really registered
        assert "/code-review" not in _READONLY_ALLOWED_COMMANDS

        session = _RecordingSession(readonly=True)
        host = _CapturingHost()
        await handle_slash_command("/code-review do it", session, _StubClient(), host, _StubFmt())

        assert session.calls == [], (
            f"skill command posted a turn into the viewed child: {session.calls}"
        )
        assert "read-only view" in host.render_plain().lower()
    finally:
        unregister_skill_commands(registered)


async def test_navigational_command_allowed_in_readonly_view() -> None:
    """Allowlisted read-only / navigational commands are NOT blocked while
    observing a sub-agent — they don't touch the child. ``/help`` is a safe,
    observable representative (it lists the registered commands); the rest of
    the allowlist is covered by ``test_every_registered_command_is_allowlisted
    _or_blocked`` without invoking side-effecting handlers like ``/quit`` or
    ``/report``."""
    session = _RecordingSession(readonly=True)
    host = _CapturingHost()

    await handle_slash_command("/help", session, _StubClient(), host, _StubFmt())

    out = host.render_plain()
    assert "read-only view" not in out.lower()
    assert "/switch" in out  # /help dispatched and listed commands
    assert session.calls == []  # navigational commands never mutate the child


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


async def test_unknown_command_not_reported_as_readonly_block() -> None:
    """A typo'd command in read-only view still gets the 'unknown command'
    message — the guard only fires for *registered* commands, so it never
    masks an unknown-command error."""
    session = _RecordingSession(readonly=True)
    host = _CapturingHost()

    await handle_slash_command(
        "/definitely-not-a-command", session, _StubClient(), host, _StubFmt()
    )

    out = host.render_plain().lower()
    assert "read-only view" not in out
    assert "unknown command" in out
