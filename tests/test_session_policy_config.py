"""Tests for session-scoped ``--policy-config`` support.

Covers the three concerns from the feature: option parsing/validation
(:func:`load_session_policy_config` and the shared Click option),
applying the parsed policies to a session via the per-session policy API
(:func:`apply_pending_session_policies`), and that the same mechanism
fires on the shared ``bind_session_runner`` step used by both new and
resumed native launches.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import click
import httpx
import pytest
from click.testing import CliRunner

from omnigent.native import native_terminal
from omnigent.native.session_policy_config import (
    apply_pending_session_policies,
    load_session_policy_config,
    pending_session_policies,
    policy_config_option,
    prime_session_policies,
)
from omnigent.spec.types import FunctionPolicySpec, FunctionRef


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


# --- integration with the shared bind step ---------------------------------


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
async def test_bind_applies_primed_policies_on_new_session() -> None:
    """Binding a freshly launched session attaches the primed policies."""
    prime_session_policies([_spec("budget", "pkg.module.handler", None)])
    requests: list[tuple[str, bytes]] = []

    await _run_bind("conv_new", requests)

    assert ("PATCH", b"/v1/sessions/conv_new") in requests
    assert ("POST", b"/v1/sessions/conv_new/policies") in requests


@pytest.mark.asyncio
async def test_bind_applies_primed_policies_on_resume() -> None:
    """Cold-resume funnels through the same bind step, so policies apply too."""
    prime_session_policies([_spec("budget", "pkg.module.handler", None)])
    requests: list[tuple[str, bytes]] = []

    await _run_bind("conv_resumed", requests)

    assert ("POST", b"/v1/sessions/conv_resumed/policies") in requests


@pytest.mark.asyncio
async def test_bind_without_policies_only_binds() -> None:
    """Launches without --policy-config bind exactly as before (no extra POST)."""
    requests: list[tuple[str, bytes]] = []

    await _run_bind("conv_plain", requests)

    assert requests == [("PATCH", b"/v1/sessions/conv_plain")]


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
