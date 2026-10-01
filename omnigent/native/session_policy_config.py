"""Session-scoped contextual policy config for native launch commands.

``omnigent <harness> --policy-config FILE`` applies a contextual policy
config to the launched (or resumed) session ONLY, leaving server-wide
defaults untouched. The file uses the same ``policies:`` grammar as
``omnigent server --config`` — it is parsed through the identical
:func:`omnigent.spec.parse_default_policies` validator — but the parsed
policies are attached to the one session via the per-session policy CRUD
API (``POST /v1/sessions/{id}/policies``) instead of
``RuntimeCaps.default_policies``.

The parsed policies travel from the Click command to the shared
``bind_session_runner`` step through a context variable so every
per-harness launcher applies them uniformly without threading a new
parameter through each bespoke runner. The variable is primed by the
``--policy-config`` option callback (see :data:`policy_config_option`)
and consumed by :func:`apply_pending_session_policies`.

Only ``type: function`` policies are expressible (the session policy
store evaluates ``type="python"`` handlers), and the server requires each
handler to be a registered policy — a builtin or a module added via the
server's ``policy_modules`` config — so an untrusted session cannot point
a policy at an arbitrary importable callable.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    import httpx

    from omnigent.spec.types import FunctionPolicySpec

_logger = logging.getLogger(__name__)

# Policies parsed from ``--policy-config`` for the current launch. Primed by
# the option callback (CLI thread) and read inside the runner's asyncio task,
# which runs in a copy of that context, so the value propagates without a new
# parameter on every ``run_<harness>_native`` seam. Empty tuple = flag absent.
_PENDING_SESSION_POLICIES: contextvars.ContextVar[tuple[FunctionPolicySpec, ...]] = (
    contextvars.ContextVar("omnigent_pending_session_policies", default=())
)


def load_session_policy_config(path: str) -> list[FunctionPolicySpec]:
    """Load and validate a session-scoped contextual policy config file.

    Reuses the server ``--config`` policy path: the file is read as YAML and
    its ``policies:`` mapping is validated by
    :func:`omnigent.spec.parse_default_policies`, so the grammar and error
    messages match ``omnigent server --config`` exactly.

    :param path: Filesystem path to the YAML config, e.g.
        ``"./session_policies.yaml"``.
    :returns: The parsed function policies, in declaration order.
    :raises click.ClickException: If the file cannot be read, is not a YAML
        mapping, declares no policies, or declares a non-function policy
        (only function policies are session-attachable).
    """
    import yaml

    from omnigent.errors import OmnigentError
    from omnigent.spec import parse_default_policies
    from omnigent.spec.types import FunctionPolicySpec

    try:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except OSError as exc:
        raise click.ClickException(f"Could not read policy config {path!r}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise click.ClickException(f"Policy config {path!r} is not valid YAML: {exc}") from exc

    cfg = raw or {}
    if not isinstance(cfg, dict):
        raise click.ClickException(f"Policy config {path!r} must be a YAML mapping.")

    policies_raw = cfg.get("policies")
    if policies_raw is not None and not isinstance(policies_raw, dict):
        raise click.ClickException(
            f"Policy config {path!r}: `policies:` must be a mapping keyed by policy name."
        )

    try:
        specs = parse_default_policies(policies_raw)
    except OmnigentError as exc:
        raise click.ClickException(f"Policy config {path!r}: {exc}") from exc

    func_specs: list[FunctionPolicySpec] = []
    for spec in specs:
        if not isinstance(spec, FunctionPolicySpec) or spec.function is None:
            raise click.ClickException(
                f"Policy config {path!r}: policy {spec.name!r} is not a function policy; "
                "only `type: function` policies can be attached to a session."
            )
        func_specs.append(spec)

    if not func_specs:
        raise click.ClickException(
            f"Policy config {path!r} declares no policies under a top-level `policies:` mapping."
        )
    return func_specs


def prime_session_policies(specs: Sequence[FunctionPolicySpec]) -> None:
    """Record the policies to attach to the next launched/resumed session.

    :param specs: Function policies parsed from ``--policy-config``; pass an
        empty sequence to clear any previously primed policies.
    :returns: None.
    """
    _PENDING_SESSION_POLICIES.set(tuple(specs))


def pending_session_policies() -> tuple[FunctionPolicySpec, ...]:
    """Return the policies primed for the current launch.

    :returns: The primed function policies, or an empty tuple when
        ``--policy-config`` was not supplied.
    """
    return _PENDING_SESSION_POLICIES.get()


async def apply_pending_session_policies(
    client: httpx.AsyncClient,
    session_id: str,
    *,
    notify: Callable[[str], None] | None = None,
) -> int:
    """Attach any primed session-scoped policies to *session_id*.

    Called from the shared ``bind_session_runner`` step so every native
    harness applies ``--policy-config`` uniformly. A no-op (returns ``0``)
    when no policies were primed, so launches without the flag are
    unaffected. Each policy is created via ``POST
    /v1/sessions/{id}/policies``; an existing policy of the same name
    (HTTP 409) is left in place, which makes repeated resumes idempotent.

    :param client: HTTP client pointed at the Omnigent server (the daemon
        client used by the native launch flow).
    :param session_id: The session to attach policies to, e.g.
        ``"conv_abc123"``.
    :param notify: Optional sink for a one-line user-facing summary, e.g.
        ``lambda msg: click.echo(msg, err=True)``.
    :returns: The number of policies newly created on the session.
    :raises click.ClickException: If the server rejects a policy for any
        reason other than a name conflict.
    """
    import urllib.parse

    from omnigent.host.daemon_launch import error_text

    specs = _PENDING_SESSION_POLICIES.get()
    if not specs:
        return 0

    # Consume the pending policies up front: a session's policies are applied
    # exactly once per launch. Clearing here (a) stops a second bind in the same
    # context from re-applying them, and (b) lets ``ensure_session_policies_applied``
    # detect a launch that never reached this apply step — a live-terminal
    # reattach, where bind is skipped — by seeing the list still populated.
    _PENDING_SESSION_POLICIES.set(())

    encoded = urllib.parse.quote(session_id, safe="")
    created = 0
    for spec in specs:
        function = spec.function
        if function is None:  # Guarded at load time; defensive for direct callers.
            continue
        body: dict[str, object] = {
            "name": spec.name,
            "type": "python",
            "handler": function.path,
        }
        if function.arguments is not None:
            body["factory_params"] = function.arguments
        resp = await client.post(f"/v1/sessions/{encoded}/policies", json=body)
        if resp.status_code == 409:
            _logger.info(
                "session policy %r already present on %s; leaving it unchanged",
                spec.name,
                session_id,
            )
            continue
        if resp.status_code >= 400:
            raise click.ClickException(
                f"Failed to apply session policy {spec.name!r} "
                f"({resp.status_code}): {error_text(resp)}"
            )
        created += 1

    if notify is not None:
        notify(
            f"Applied {len(specs)} contextual {_plural(len(specs), 'policy', 'policies')} "
            f"to this session ({created} new)."
        )
    _logger.info(
        "applied %d session policy spec(s) to %s (%d newly created)",
        len(specs),
        session_id,
        created,
    )
    return created


def ensure_session_policies_applied() -> None:
    """Fail if ``--policy-config`` policies were primed but never applied.

    Every new or cold-resumed launch attaches its policies in
    :func:`apply_pending_session_policies` — invoked from the runner-bind step
    (``launch_or_reuse_daemon_runner`` on the daemon path, or
    ``bind_session_runner``) — which consumes the pending list. Reaching the
    post-prepare handoff with policies still pending therefore means the launch
    reused an already-live session (its runner/terminal was still running, so
    no fresh bind happened), so the requested policy was never applied.
    Changing a session's contextual policy mid-run is out of scope (tracked
    follow-up), so fail loudly here rather than silently drop the flag.

    Called from each native launcher right before the "Web UI" handoff. A
    no-op when nothing was primed (the common case) or when apply already
    consumed the list.

    :returns: None.
    :raises click.ClickException: When policies were primed but not applied.
    """
    pending = _PENDING_SESSION_POLICIES.get()
    if not pending:
        return
    names = ", ".join(sorted(spec.name for spec in pending))
    # Consume so a retry within the same context (embedded/reentrant use)
    # starts clean rather than re-raising on a stale list.
    _PENDING_SESSION_POLICIES.set(())
    raise click.ClickException(
        "--policy-config cannot be applied here: this resumes a session that is "
        "already running (its runner/terminal is live), and changing a session's "
        "contextual policy mid-run is not supported yet (tracked follow-up). Stop "
        "the running session first, or relaunch without --policy-config. "
        f"Policies not applied: {names}."
    )


def _plural(count: int, singular: str, plural: str) -> str:
    """Return *singular* when *count* is 1, else *plural*.

    :param count: The quantity being described.
    :param singular: Word to use for a count of one.
    :param plural: Word to use otherwise.
    :returns: The number-appropriate word.
    """
    return singular if count == 1 else plural


def _policy_config_callback(
    ctx: click.Context,
    param: click.Parameter,
    value: str | None,
) -> str | None:
    """Click callback that primes session policies from ``--policy-config``.

    Primes (or clears) the pending-policy context variable as a side effect
    so the shared launch path can attach the policies once the session id is
    known, without threading the value through each harness runner. Runs with
    ``expose_value=False`` so no launcher signature changes.

    :param ctx: The active Click context (unused).
    :param param: The option parameter (unused).
    :param value: The config path, or ``None`` when the flag is absent.
    :returns: *value*, unchanged.
    :raises click.ClickException: If the referenced config is invalid.
    """
    del ctx, param
    prime_session_policies(load_session_policy_config(value) if value else [])
    return value


#: Shared ``--policy-config`` option for the per-harness launch commands.
#: Applied uniformly so ``omnigent pi``, ``omnigent codex``, and every other
#: ``omnigent <harness>`` command accept the same session-scoped policy flag.
policy_config_option = click.option(
    "--policy-config",
    "policy_config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    expose_value=False,
    callback=_policy_config_callback,
    help=(
        "Apply a session-scoped contextual policy config (a YAML file with a "
        "top-level `policies:` mapping, same grammar as `omnigent server "
        "--config`) to THIS session only; server-wide defaults are unchanged. "
        "Applied on launch and on cold resume (a session that is not currently "
        "running); resuming a session that is already live fails rather than "
        "changing policy mid-run. Handlers must be registered policies "
        "(builtins, or modules added via the server's `policy_modules`)."
    ),
)
