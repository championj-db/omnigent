"""Tests for session-scoped ``--policy-config`` support.

Covers the three concerns from the feature: option parsing/validation
(:func:`load_session_policy_config` and the shared Click option),
applying the parsed policies to a session via the per-session policy API
(:func:`apply_pending_session_policies`), and that the same mechanism
fires on the shared ``bind_session_runner`` step used by both new and
resumed native launches.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import click
import httpx
import pytest
from click.testing import CliRunner

from omnigent.host.daemon_launch import launch_or_reuse_daemon_runner
from omnigent.native import native_terminal
from omnigent.native.session_policy_config import (
    apply_pending_session_policies,
    ensure_session_policies_applied,
    load_session_policy_config,
    pending_session_policies,
    policy_config_option,
    prime_session_policies,
)
from omnigent.spec.types import FunctionPolicySpec, FunctionRef

# Harness commands that must all accept --policy-config (parameterized test #5).
_HARNESS_COMMANDS = (
    "claude",
    "codex",
    "cursor",
    "pi",
    "goose",
    "kimi",
    "qwen",
    "antigravity",
    "devin",
    "kiro",
    "hermes",
    "opencode",
)


@pytest.fixture(autouse=True)
def _reset_pending_policies() -> Iterator[None]:
    """Clear primed policies around every test to prevent cross-test leakage.

    :returns: Iterator yielding once, resetting before and after the test.
    """
    prime_session_policies([])
    yield
    prime_session_policies([])


def _write_config(tmp_path: Path, body: str) -> str:
    """Write *body* to a temp YAML file and return its path.

    :param tmp_path: pytest temp directory.
    :param body: YAML text to write.
    :returns: The written file path as a string.
    """
    path = tmp_path / "policies.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


_VALID_CONFIG = """\
policies:
  session_budget:
    type: function
    handler: omnigent.policies.builtins.cost.cost_budget
    factory_params:
      max_cost_usd: 10.0
  rate_limit:
    type: function
    handler: omnigent.policies.builtins.safety.max_tool_calls_per_session
"""


# --- load_session_policy_config -------------------------------------------


def test_load_parses_function_policies_with_and_without_params(tmp_path: Path) -> None:
    """A valid config yields function specs mapping handler + factory params."""
    specs = load_session_policy_config(_write_config(tmp_path, _VALID_CONFIG))

    assert [s.name for s in specs] == ["session_budget", "rate_limit"]
    first = specs[0]
    assert isinstance(first, FunctionPolicySpec)
    assert first.function is not None
    assert first.function.path == "omnigent.policies.builtins.cost.cost_budget"
    assert first.function.arguments == {"max_cost_usd": 10.0}
    # A policy without factory_params carries no arguments.
    assert specs[1].function is not None
    assert specs[1].function.arguments is None


def test_load_rejects_empty_policies(tmp_path: Path) -> None:
    """A file with no policies is a user error, not a silent no-op."""
    with pytest.raises(click.ClickException, match="no policies"):
        load_session_policy_config(_write_config(tmp_path, "policies: {}\n"))


def test_load_rejects_missing_policies_key(tmp_path: Path) -> None:
    """A config whose only keys are unrelated still has no policies to apply."""
    with pytest.raises(click.ClickException, match="no policies"):
        load_session_policy_config(_write_config(tmp_path, "llm:\n  model: x\n"))


def test_load_rejects_non_mapping_policies(tmp_path: Path) -> None:
    """``policies:`` must be a mapping keyed by name."""
    with pytest.raises(click.ClickException, match="must be a mapping"):
        load_session_policy_config(_write_config(tmp_path, "policies:\n  - a\n  - b\n"))


def test_load_rejects_non_mapping_document(tmp_path: Path) -> None:
    """A top-level non-mapping YAML document is rejected."""
    with pytest.raises(click.ClickException, match="must be a YAML mapping"):
        load_session_policy_config(_write_config(tmp_path, "- just\n- a\n- list\n"))


def test_load_surfaces_parser_validation_error(tmp_path: Path) -> None:
    """Malformed policy entries surface the shared parser's message."""
    bad = "policies:\n  broken:\n    type: prompt\n    prompt: hi\n"
    with pytest.raises(click.ClickException, match="prompt"):
        load_session_policy_config(_write_config(tmp_path, bad))


def test_load_rejects_invalid_yaml(tmp_path: Path) -> None:
    """Invalid YAML is reported as such, not as a stack trace."""
    with pytest.raises(click.ClickException, match="not valid YAML"):
        load_session_policy_config(_write_config(tmp_path, "policies: [unclosed\n"))


# --- shared --policy-config option parsing ---------------------------------


def _probe_command() -> click.Command:
    """Build a minimal command carrying only the shared policy option.

    :returns: A Click command that records the primed policy names.
    """

    @click.command()
    @policy_config_option
    def probe() -> None:
        """Echo the primed policy names as JSON."""
        click.echo(json.dumps([s.name for s in pending_session_policies()]))

    return probe


def test_option_primes_policies_when_passed(tmp_path: Path) -> None:
    """``--policy-config FILE`` parses and primes the policies during parsing."""
    cfg = _write_config(tmp_path, _VALID_CONFIG)
    result = CliRunner().invoke(_probe_command(), ["--policy-config", cfg])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == ["session_budget", "rate_limit"]


def test_option_absent_primes_nothing(tmp_path: Path) -> None:
    """Omitting the flag leaves the pending policies empty (backward compatible)."""
    result = CliRunner().invoke(_probe_command(), [])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []


def test_option_rejects_missing_file() -> None:
    """A nonexistent path fails during parsing via ``click.Path(exists=True)``."""
    result = CliRunner().invoke(_probe_command(), ["--policy-config", "/no/such/file.yaml"])

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_option_reports_invalid_config(tmp_path: Path) -> None:
    """An existing-but-invalid config fails parsing with a readable message."""
    cfg = _write_config(tmp_path, "policies: {}\n")
    result = CliRunner().invoke(_probe_command(), ["--policy-config", cfg])

    assert result.exit_code != 0
    assert "no policies" in result.output


@pytest.mark.parametrize("command", _HARNESS_COMMANDS)
def test_every_harness_command_exposes_policy_config(command: str) -> None:
    """`--policy-config` is wired uniformly onto every `omni <harness>` command."""
    from omnigent.cli import cli

    result = CliRunner().invoke(cli, [command, "--help"])

    assert result.exit_code == 0, result.output
    assert "--policy-config" in result.output


# --- apply_pending_session_policies ----------------------------------------


def _spec(name: str, path: str, arguments: dict[str, object] | None) -> FunctionPolicySpec:
    """Construct a function policy spec for apply-path tests.

    :param name: Policy name.
    :param path: Handler dotted path.
    :param arguments: Optional factory params.
    :returns: The spec.
    """
    return FunctionPolicySpec(
        name=name, on=None, function=FunctionRef(path=path, arguments=arguments)
    )


@pytest.mark.asyncio
async def test_apply_is_noop_when_nothing_primed() -> None:
    """With no primed policies, apply makes no request and returns 0."""
    calls: list[httpx.Request] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        """Record any unexpected outbound request."""
        calls.append(request)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(
        base_url="https://example.databricks.com",
        transport=httpx.MockTransport(_handler),
    ) as client:
        created = await apply_pending_session_policies(client, "conv_abc")

    assert created == 0
    assert calls == []


@pytest.mark.asyncio
async def test_apply_posts_each_policy_with_mapped_body() -> None:
    """Each primed policy POSTs type=python + handler + factory_params."""
    prime_session_policies(
        [
            _spec("budget", "omnigent.policies.builtins.cost.cost_budget", {"max_cost_usd": 5.0}),
            _spec("rate", "omnigent.policies.builtins.safety.max_tool_calls_per_session", None),
        ]
    )
    seen: list[dict[str, object]] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        """Capture POSTs to the session policies route."""
        assert request.method == "POST"
        assert request.url.raw_path == b"/v1/sessions/conv_abc/policies"
        seen.append(json.loads(request.read()))
        return httpx.Response(200, json={"id": "pol_x"})

    notes: list[str] = []
    async with httpx.AsyncClient(
        base_url="https://example.databricks.com",
        transport=httpx.MockTransport(_handler),
    ) as client:
        created = await apply_pending_session_policies(client, "conv_abc", notify=notes.append)

    assert created == 2
    assert seen == [
        {
            "name": "budget",
            "type": "python",
            "handler": "omnigent.policies.builtins.cost.cost_budget",
            "factory_params": {"max_cost_usd": 5.0},
        },
        {
            "name": "rate",
            "type": "python",
            "handler": "omnigent.policies.builtins.safety.max_tool_calls_per_session",
        },
    ]
    assert notes and "2" in notes[0]


@pytest.mark.asyncio
async def test_apply_encodes_session_id_in_path() -> None:
    """Session ids with path separators are percent-encoded."""
    prime_session_policies([_spec("p", "pkg.module.handler", None)])
    seen: dict[str, object] = {}

    async def _handler(request: httpx.Request) -> httpx.Response:
        """Capture the encoded request path."""
        seen["path"] = request.url.raw_path
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(
        base_url="https://example.databricks.com",
        transport=httpx.MockTransport(_handler),
    ) as client:
        await apply_pending_session_policies(client, "conv/a b")

    assert seen["path"] == b"/v1/sessions/conv%2Fa%20b/policies"


@pytest.mark.asyncio
async def test_apply_treats_conflict_as_already_present() -> None:
    """A 409 (duplicate name) is skipped so repeated resumes are idempotent."""
    prime_session_policies(
        [
            _spec("dup", "pkg.module.handler", None),
            _spec("new", "pkg.module.other", None),
        ]
    )
    statuses = iter([409, 200])

    async def _handler(request: httpx.Request) -> httpx.Response:
        """Return 409 for the first policy, 200 for the second."""
        return httpx.Response(next(statuses), json={"detail": "exists"})

    async with httpx.AsyncClient(
        base_url="https://example.databricks.com",
        transport=httpx.MockTransport(_handler),
    ) as client:
        created = await apply_pending_session_policies(client, "conv_abc")

    # Both were sent; only the non-conflicting one counts as newly created.
    assert created == 1


@pytest.mark.asyncio
async def test_apply_raises_on_server_rejection() -> None:
    """A non-conflict 4xx surfaces as a user-facing ClickException."""
    prime_session_policies([_spec("bad", "pkg.module.handler", None)])

    async def _handler(request: httpx.Request) -> httpx.Response:
        """Reject the policy as unregistered."""
        return httpx.Response(400, json={"detail": "handler not registered"}, request=request)

    async with httpx.AsyncClient(
        base_url="https://example.databricks.com",
        transport=httpx.MockTransport(_handler),
    ) as client:
        with pytest.raises(click.ClickException, match="handler not registered"):
            await apply_pending_session_policies(client, "conv_abc")


# --- atomicity: policies POSTed before the runner bind (bind_session_runner) ---


async def _run_bind(session_id: str, requests: list[tuple[str, bytes]]) -> None:
    """Call ``bind_session_runner`` against a recording transport.

    :param session_id: Session id to bind.
    :param requests: Sink of ``(method, raw_path)`` tuples per request.
    :returns: None.
    """

    async def _handler(request: httpx.Request) -> httpx.Response:
        """Record each request and answer success."""
        requests.append((request.method, request.url.raw_path))
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(
        base_url="https://example.databricks.com",
        transport=httpx.MockTransport(_handler),
    ) as client:
        await native_terminal.bind_session_runner(client, session_id, "runner_1")


@pytest.mark.asyncio
async def test_bind_posts_policies_before_binding_runner() -> None:
    """Policies are POSTed BEFORE the runner-bind PATCH (atomic ordering)."""
    prime_session_policies([_spec("budget", "pkg.module.handler", None)])
    requests: list[tuple[str, bytes]] = []

    await _run_bind("conv_new", requests)

    # POST /policies strictly precedes the PATCH that binds the runner.
    assert requests == [
        ("POST", b"/v1/sessions/conv_new/policies"),
        ("PATCH", b"/v1/sessions/conv_new"),
    ]


@pytest.mark.asyncio
async def test_bind_aborts_without_binding_when_policy_rejected() -> None:
    """A rejected policy aborts the launch with no runner-bind PATCH sent."""
    prime_session_policies([_spec("bad", "pkg.module.handler", None)])
    requests: list[tuple[str, bytes]] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        """Reject the policy POST; record every request."""
        requests.append((request.method, request.url.raw_path))
        if request.url.path.endswith("/policies"):
            return httpx.Response(400, json={"detail": "handler not registered"}, request=request)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(
        base_url="https://example.databricks.com",
        transport=httpx.MockTransport(_handler),
    ) as client:
        with pytest.raises(click.ClickException, match="handler not registered"):
            await native_terminal.bind_session_runner(client, "conv_new", "runner_1")

    # Only the (failed) policy POST was attempted — the runner was never bound.
    assert requests == [("POST", b"/v1/sessions/conv_new/policies")]


@pytest.mark.asyncio
async def test_bind_without_policies_only_binds() -> None:
    """Launches without --policy-config bind exactly as before (no extra POST)."""
    requests: list[tuple[str, bytes]] = []

    await _run_bind("conv_plain", requests)

    assert requests == [("PATCH", b"/v1/sessions/conv_plain")]


# --- live-reattach guard ----------------------------------------------------


def test_guard_is_noop_when_nothing_pending() -> None:
    """The handoff guard does nothing when --policy-config was not supplied."""
    ensure_session_policies_applied()  # no raise


def test_guard_raises_when_policies_were_never_applied() -> None:
    """Pending-but-unapplied policies (live reattach) fail with a clear error."""
    prime_session_policies([_spec("budget", "pkg.module.handler", None)])

    with pytest.raises(click.ClickException, match="already running") as excinfo:
        ensure_session_policies_applied()

    assert "mid-run is not supported" in str(excinfo.value)
    # Consumed on raise so a retry in the same context starts clean.
    assert pending_session_policies() == ()


# --- genuine daemon flow (launch_or_reuse_daemon_runner via asyncio.run) -----
#
# These drive the universal daemon bind point end to end THROUGH ``asyncio.run``,
# so they prove the context variable primed on the CLI thread survives the real
# async boundary into the runner coroutine, exactly as a native launch does.


def _daemon_handler(
    requests: list[tuple[str, bytes]],
    *,
    bound_runner: str | None,
    runner_online: bool,
) -> httpx.MockTransport:
    """Build a MockTransport emulating the daemon runner-launch endpoints.

    :param requests: Sink of ``(method, raw_path)`` per request.
    :param bound_runner: Runner id already bound to the session, or ``None``.
    :param runner_online: Whether that bound runner reports online.
    :returns: A configured :class:`httpx.MockTransport`.
    """

    async def _handler(request: httpx.Request) -> httpx.Response:
        """Answer the GET session / runner-status / launch / policy routes."""
        requests.append((request.method, request.url.raw_path))
        path = request.url.path
        if request.method == "GET" and path.startswith("/v1/sessions/"):
            body = {"runner_id": bound_runner} if bound_runner else {}
            return httpx.Response(200, json=body)
        if path.endswith("/status"):
            return httpx.Response(200, json={"online": runner_online})
        if path.endswith("/policies"):
            return httpx.Response(200, json={"id": "pol_x"})
        if path.endswith("/runners"):
            return httpx.Response(200, json={"runner_id": "runner_fresh"})
        return httpx.Response(200, json={})

    return httpx.MockTransport(_handler)


def _run_launch(transport: httpx.MockTransport, *, fresh: bool) -> str:
    """Prime policies, then drive ``launch_or_reuse_daemon_runner`` via asyncio.run.

    Priming happens on this (synchronous) thread; ``asyncio.run`` copies the
    current context into the coroutine, mirroring how a native launcher primes
    in its Click callback and then calls ``asyncio.run(_drive())``.

    :param transport: Mock transport for the daemon endpoints.
    :param fresh: The ``fresh`` flag (``True`` for a brand-new session).
    :returns: The bound runner id.
    """

    async def _drive() -> str:
        async with httpx.AsyncClient(
            base_url="https://example.databricks.com", transport=transport
        ) as client:
            return await launch_or_reuse_daemon_runner(
                client,
                host_id="host_1",
                session_id="conv_x",
                workspace="/w",
                fresh=fresh,
            )

    return asyncio.run(_drive())


def test_daemon_fresh_launch_applies_policies_across_asyncio_boundary() -> None:
    """A brand-new daemon launch POSTs the primed policies before launching."""
    prime_session_policies([_spec("budget", "pkg.module.handler", None)])
    requests: list[tuple[str, bytes]] = []
    transport = _daemon_handler(requests, bound_runner=None, runner_online=False)

    runner_id = _run_launch(transport, fresh=True)

    assert runner_id == "runner_fresh"
    methods = [(m, p) for (m, p) in requests]
    assert ("POST", b"/v1/sessions/conv_x/policies") in methods
    # Policy POST precedes the runner launch (atomic bind ordering).
    assert methods.index(("POST", b"/v1/sessions/conv_x/policies")) < methods.index(
        ("POST", b"/v1/hosts/host_1/runners")
    )


def test_daemon_cold_resume_applies_policies() -> None:
    """Resuming a session with no live runner reaches the apply hook and POSTs."""
    prime_session_policies([_spec("budget", "pkg.module.handler", None)])
    requests: list[tuple[str, bytes]] = []
    # No bound runner → cold resume → fresh launch path → policies applied.
    transport = _daemon_handler(requests, bound_runner=None, runner_online=False)

    runner_id = _run_launch(transport, fresh=False)

    assert runner_id == "runner_fresh"
    assert ("POST", b"/v1/sessions/conv_x/policies") in requests


def test_daemon_live_reattach_skips_apply_and_guard_fails() -> None:
    """Reusing an online runner (live reattach) applies nothing; the guard fails."""
    prime_session_policies([_spec("budget", "pkg.module.handler", None)])
    requests: list[tuple[str, bytes]] = []
    # Session already bound to an ONLINE runner → reuse early-return, no launch.
    transport = _daemon_handler(requests, bound_runner="runner_live", runner_online=True)

    runner_id = _run_launch(transport, fresh=False)

    assert runner_id == "runner_live"
    # No policy POST happened — the online-runner reuse returned early.
    assert all(not p.endswith(b"/policies") for (_m, p) in requests)
    # Policies remain pending, so the launcher's handoff guard fails clearly.
    with pytest.raises(click.ClickException, match="already running"):
        ensure_session_policies_applied()


def test_applied_state_survives_asyncio_run_boundary() -> None:
    """A successful launch's apply is visible to the SYNC hand-off guard (B1).

    Claude's guards run synchronously *after* ``asyncio.run()`` returns. The
    apply happens inside that coroutine; with a ``ContextVar`` the clear would
    not propagate out and the guard would falsely reject a successful launch.
    The module global propagates, so the guard sees the applied state.
    """
    prime_session_policies([_spec("budget", "pkg.module.handler", None)])
    requests: list[tuple[str, bytes]] = []
    transport = _daemon_handler(requests, bound_runner=None, runner_online=False)

    # Apply happens inside the coroutine driven by asyncio.run (copied context).
    runner_id = _run_launch(transport, fresh=True)
    assert runner_id == "runner_fresh"
    assert ("POST", b"/v1/sessions/conv_x/policies") in requests

    # The guard now runs synchronously, as a native launcher's does, and must
    # NOT raise — it sees the policies as applied (cleared), not pending.
    ensure_session_policies_applied()


# --- claude daemon resume: online runner, terminal liveness decision ---------
#
# Claude's daemon path reuses an online runner without the fresh-launch policy
# attach, so ``_attach_policy_config_for_resume`` decides from terminal liveness
# and must FAIL CLOSED on any non-authoritative outcome (B1), and do NO work
# when the flag is absent (B2).


def _claude_resume_handler(
    requests: list[tuple[str, bytes]],
    *,
    terminal_status: int = 200,
    terminal_running: bool | None = True,
    terminal_exc: type[httpx.TransportError] | None = None,
) -> httpx.MockTransport:
    """Build a transport driving the claude terminal-liveness GET + policy POST.

    :param requests: Sink of ``(method, raw_path)`` per request.
    :param terminal_status: Status for the terminal GET (200/404/409/5xx/...).
    :param terminal_running: For a 200, the ``metadata.running`` value; ``None``
        omits the metadata entirely (shape = live-but-unspecified).
    :param terminal_exc: When set, the terminal GET raises this transport error
        (timeout / connection error) instead of returning a response.
    :returns: A configured :class:`httpx.MockTransport`.
    """
    from omnigent.harnesses.claude_native.main import claude_terminal_resource_id

    terminal_id = claude_terminal_resource_id()

    async def _handler(request: httpx.Request) -> httpx.Response:
        """Answer the terminal-liveness GET and the policy POST."""
        requests.append((request.method, request.url.raw_path))
        if "/resources/terminals/" in request.url.path:
            if terminal_exc is not None:
                raise terminal_exc("boom", request=request)
            if terminal_status == 200:
                body: dict[str, object] = {"id": terminal_id, "type": "terminal"}
                if terminal_running is not None:
                    body["metadata"] = {"running": terminal_running}
                return httpx.Response(200, json=body)
            return httpx.Response(terminal_status, json={"detail": "x"}, request=request)
        if request.url.path.endswith("/policies"):
            return httpx.Response(200, json={"id": "pol_x"})
        return httpx.Response(200, json={})

    return httpx.MockTransport(_handler)


def _run_claude_resume(transport: httpx.MockTransport) -> None:
    """Drive ``_attach_policy_config_for_resume`` through asyncio.run.

    :param transport: Mock transport for the claude resume endpoints.
    :returns: None.
    """
    from omnigent.harnesses.claude_native.main import _attach_policy_config_for_resume

    async def _drive() -> None:
        async with httpx.AsyncClient(
            base_url="https://example.databricks.com", transport=transport
        ) as client:
            await _attach_policy_config_for_resume(client, "conv_x")

    asyncio.run(_drive())


def test_claude_resume_absent_terminal_applies_and_passes_guard() -> None:
    """(a) 404 (torn-down) terminal → policy applied; hand-off guard passes."""
    prime_session_policies([_spec("budget", "pkg.module.handler", None)])
    requests: list[tuple[str, bytes]] = []

    _run_claude_resume(_claude_resume_handler(requests, terminal_status=404))

    # Authoritative not-found → restart → policy IS applied before relaunch.
    assert ("POST", b"/v1/sessions/conv_x/policies") in requests
    # Applied → the synchronous hand-off guard does not raise.
    ensure_session_policies_applied()


def test_claude_resume_stopped_terminal_applies() -> None:
    """A 200 with ``running=False`` is authoritative absent → policy applied."""
    prime_session_policies([_spec("budget", "pkg.module.handler", None)])
    requests: list[tuple[str, bytes]] = []

    _run_claude_resume(_claude_resume_handler(requests, terminal_running=False))

    assert ("POST", b"/v1/sessions/conv_x/policies") in requests
    ensure_session_policies_applied()


def test_claude_resume_live_terminal_fails_closed() -> None:
    """A genuinely live terminal (200 running) applies nothing and raises."""
    prime_session_policies([_spec("budget", "pkg.module.handler", None)])
    requests: list[tuple[str, bytes]] = []

    with pytest.raises(click.ClickException, match="already running"):
        _run_claude_resume(_claude_resume_handler(requests, terminal_running=True))

    assert all(not p.endswith(b"/policies") for (_m, p) in requests)
    assert pending_session_policies() == ()  # consumed on raise


@pytest.mark.parametrize("status", [500, 502, 503])
def test_claude_resume_server_error_fails_closed(status: int) -> None:
    """(b) A 5xx while probing a maybe-live terminal fails closed — no relaunch."""
    prime_session_policies([_spec("budget", "pkg.module.handler", None)])
    requests: list[tuple[str, bytes]] = []

    with pytest.raises(click.ClickException, match="already running"):
        _run_claude_resume(_claude_resume_handler(requests, terminal_status=status))

    assert all(not p.endswith(b"/policies") for (_m, p) in requests)


@pytest.mark.parametrize("exc", [httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError])
def test_claude_resume_transport_error_fails_closed(exc: type[httpx.TransportError]) -> None:
    """(c) A timeout / connection error fails closed — no apply, raises."""
    prime_session_policies([_spec("budget", "pkg.module.handler", None)])
    requests: list[tuple[str, bytes]] = []

    with pytest.raises(click.ClickException, match="already running"):
        _run_claude_resume(_claude_resume_handler(requests, terminal_exc=exc))

    assert all(not p.endswith(b"/policies") for (_m, p) in requests)


@pytest.mark.parametrize("status", [409, 418])
def test_claude_resume_non_404_status_fails_closed(status: int) -> None:
    """(d) 409 / other non-404 status is not authoritative absent → fail closed."""
    prime_session_policies([_spec("budget", "pkg.module.handler", None)])
    requests: list[tuple[str, bytes]] = []

    with pytest.raises(click.ClickException, match="already running"):
        _run_claude_resume(_claude_resume_handler(requests, terminal_status=status))

    assert all(not p.endswith(b"/policies") for (_m, p) in requests)


def test_claude_resume_without_policy_makes_no_request() -> None:
    """B2: a policy-free claude resume issues NO liveness request (inert path)."""
    prime_session_policies([])  # flag absent → nothing pending

    def _boom(request: httpx.Request) -> httpx.Response:
        """Fail if any HTTP request is attempted."""
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    # Must early-return before touching the transport; must not raise.
    _run_claude_resume(httpx.MockTransport(_boom))


# --- real native command wiring (option parsing on `omnigent pi`) ----------


def _stub_pi_launch(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Stub ``omnigent pi``'s heavy launch deps and capture the runner call.

    Replaces the daemon/server-touching helpers so the command runs to the
    point of invoking ``run_pi_native``, where it records the kwargs and the
    policies primed at that moment.

    :param monkeypatch: pytest patcher.
    :returns: A list that receives one dict per ``run_pi_native`` call.
    """
    import omnigent.cli as cli_mod
    import omnigent.harness_startup_config as startup_mod
    import omnigent.harnesses.pi_native.main as pi_main

    monkeypatch.setattr(cli_mod, "_load_effective_config", dict, raising=True)
    monkeypatch.setattr(cli_mod, "_ensure_backend", lambda server: server or "http://local", True)
    monkeypatch.setattr(
        cli_mod, "_resolve_auto_open_conversation_from_config", lambda cfg: False, raising=True
    )
    monkeypatch.setattr(
        cli_mod, "_resolve_harness_startup_args", lambda cfg, harness, args: tuple(args), True
    )
    monkeypatch.setattr(
        startup_mod,
        "resolve_harness_command",
        lambda harness, default="", explicit=None, cfg=None: "",
        raising=True,
    )

    calls: list[dict[str, object]] = []

    def _fake_run_pi_native(**kwargs: object) -> None:
        """Record the launch kwargs and the policies primed at call time."""
        kwargs["primed"] = [s.name for s in pending_session_policies()]
        calls.append(kwargs)

    monkeypatch.setattr(pi_main, "run_pi_native", _fake_run_pi_native, raising=True)
    return calls


def test_pi_command_primes_policies_for_new_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``omnigent pi --policy-config FILE`` primes the policies before launch."""
    from omnigent.cli import cli

    calls = _stub_pi_launch(monkeypatch)
    cfg = _write_config(tmp_path, _VALID_CONFIG)

    result = CliRunner().invoke(cli, ["pi", "--policy-config", cfg])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["session_id"] is None  # fresh session
    assert calls[0]["primed"] == ["session_budget", "rate_limit"]


def test_pi_command_primes_policies_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--policy-config`` composes with ``--resume <id>`` on a real command."""
    from omnigent.cli import cli

    calls = _stub_pi_launch(monkeypatch)
    cfg = _write_config(tmp_path, _VALID_CONFIG)

    result = CliRunner().invoke(cli, ["pi", "--resume", "conv_abc123", "--policy-config", cfg])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["session_id"] == "conv_abc123"  # resume target
    assert calls[0]["primed"] == ["session_budget", "rate_limit"]
