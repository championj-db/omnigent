"""Import native harness MCP servers and skills into Omnigent bundles."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import tomllib
import yaml

from omnigent.errors import ErrorCode, OmnigentError
from omnigent.spec import load as load_spec

ImportSource = Literal["claude", "codex", "cursor", "pi"]
ConflictMode = Literal["fail", "skip", "overwrite", "prompt"]
ImportKind = Literal["mcp", "skill", "scaffold", "env", "provenance"]

_ENV_REF_RE = re.compile(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$")
_EMBEDDED_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")
_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|auth|bearer|client[_-]?secret|credential|password|secret|token)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"^(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{20,}|xox[baprs]-|ya29\.|"
    r"[A-Za-z0-9_=-]{32,})"
)


@dataclass(frozen=True)
class PlannedArtifact:
    """One file or directory the import will create or update."""

    kind: ImportKind
    relative_path: str
    action: Literal["create", "update", "skip", "unchanged"]
    content: str | None = None
    source_dir: Path | None = None
    warning: str | None = None


@dataclass(frozen=True)
class ImportPlan:
    """Computed import result before or after applying writes."""

    source: ImportSource
    source_dir: Path
    into: Path
    dry_run: bool
    artifacts: list[PlannedArtifact] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def changed_artifacts(self) -> list[PlannedArtifact]:
        """Artifacts that would change on disk."""

        return [a for a in self.artifacts if a.action in {"create", "update"}]


@dataclass(frozen=True)
class DriftResult:
    """Result of ``omnigent agent import <source> --check``."""

    imports_path: Path
    missing: list[str]
    changed: list[str]
    ok: list[str]

    @property
    def clean(self) -> bool:
        """Return whether all recorded artifacts match provenance."""

        return not self.missing and not self.changed


PromptResolver = Callable[[str, Path], ConflictMode]


def default_source_dir(source: ImportSource) -> Path:
    """Return the default native config directory for *source*."""

    if source == "claude":
        return Path.home() / ".claude"
    if source == "codex":
        return Path.home() / ".codex"
    if source == "cursor":
        return Path.home() / ".cursor"
    if source == "pi":
        return Path.home() / ".pi"
    raise AssertionError(source)


def plan_import(
    *,
    source: ImportSource,
    source_dir: Path | None,
    into: Path,
    import_mcp: bool,
    import_skills: bool,
    dry_run: bool,
    on_conflict: ConflictMode,
    force: bool = False,
    prompt_resolver: PromptResolver | None = None,
) -> ImportPlan:
    """Build an import plan and validate it against the existing parser."""

    if source == "pi" and source_dir is None:
        raise OmnigentError(
            "Pi is not an import source unless --from points at a concrete directory.",
            code=ErrorCode.INVALID_INPUT,
        )
    resolved_source_dir = (source_dir or default_source_dir(source)).expanduser().resolve()
    into = into.expanduser().resolve()
    if not import_mcp and not import_skills:
        import_mcp = True

    warnings: list[str] = []
    artifacts: list[PlannedArtifact] = []
    env_example: dict[str, str] = {}

    if not (into / "config.yaml").exists():
        artifacts.extend(_scaffold_artifacts(source, into, on_conflict, prompt_resolver))

    if import_mcp:
        servers = _discover_mcp_servers(source, resolved_source_dir, warnings)
        if not servers:
            warnings.append(f"No MCP servers found in {resolved_source_dir}.")
        for server in servers:
            content = _render_mcp_yaml(server, env_example, force=force, warnings=warnings)
            rel = f"tools/mcp/{server.name}.yaml"
            artifacts.append(
                _planned_file(
                    into,
                    rel,
                    content,
                    kind="mcp",
                    on_conflict=on_conflict,
                    prompt_resolver=prompt_resolver,
                )
            )

    if import_skills:
        skills = _discover_skill_dirs(source, resolved_source_dir)
        if not skills:
            warnings.append(f"No skills found in {resolved_source_dir}.")
        for skill in skills:
            rel = f"skills/{_safe_name(skill.name)}"
            artifacts.append(
                _planned_dir(
                    into,
                    rel,
                    skill,
                    kind="skill",
                    on_conflict=on_conflict,
                    prompt_resolver=prompt_resolver,
                )
            )

    if env_example:
        artifacts.append(
            _planned_file(
                into,
                ".env.example",
                _merge_env_example(into / ".env.example", env_example),
                kind="env",
                on_conflict="overwrite",
                prompt_resolver=prompt_resolver,
            )
        )

    artifacts = [a for a in artifacts if a.action != "skip"]
    provenance_content = _render_provenance(
        into=into,
        source=source,
        source_dir=resolved_source_dir,
        artifacts=artifacts,
        warnings=warnings,
    )
    artifacts.append(
        _planned_file(
            into,
            ".omnigent/imports.yaml",
            provenance_content,
            kind="provenance",
            on_conflict="overwrite",
            prompt_resolver=prompt_resolver,
        )
    )

    plan = ImportPlan(
        source=source,
        source_dir=resolved_source_dir,
        into=into,
        dry_run=dry_run,
        artifacts=artifacts,
        warnings=warnings,
    )
    _validate_planned_bundle(plan)
    return plan


def apply_import_plan(plan: ImportPlan) -> None:
    """Apply a previously validated import plan."""

    if plan.dry_run:
        return
    plan.into.mkdir(parents=True, exist_ok=True)
    for artifact in plan.artifacts:
        if artifact.action not in {"create", "update"}:
            continue
        target = plan.into / artifact.relative_path
        if artifact.source_dir is not None:
            if target.exists():
                _remove_path(target)
            shutil.copytree(artifact.source_dir, target)
            continue
        if artifact.content is None:
            continue
        if target.exists() and target.is_dir():
            _remove_path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(artifact.content, encoding="utf-8")


def check_import_drift(into: Path) -> DriftResult:
    """Compare current bundle artifacts against ``.omnigent/imports.yaml``."""

    into = into.expanduser().resolve()
    imports_path = into / ".omnigent" / "imports.yaml"
    if not imports_path.exists():
        raise OmnigentError(
            f"No import provenance found at {imports_path}",
            code=ErrorCode.INVALID_INPUT,
        )
    raw = yaml.safe_load(imports_path.read_text(encoding="utf-8")) or {}
    entries = raw.get("imports") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise OmnigentError(
            f"Import provenance has no imports: {imports_path}",
            code=ErrorCode.INVALID_INPUT,
        )
    artifacts: dict[str, str] = {}
    for entry in entries:
        entry_artifacts = entry.get("artifacts") if isinstance(entry, dict) else None
        if isinstance(entry_artifacts, dict):
            artifacts.update({str(k): str(v) for k, v in entry_artifacts.items()})
    if not artifacts:
        raise OmnigentError(
            f"Import provenance has no artifact hashes: {imports_path}",
            code=ErrorCode.INVALID_INPUT,
        )

    missing: list[str] = []
    changed: list[str] = []
    ok: list[str] = []
    for rel, expected in sorted(artifacts.items()):
        target = into / rel
        if not target.exists():
            missing.append(rel)
            continue
        actual = _hash_path(target)
        if actual != expected:
            changed.append(rel)
        else:
            ok.append(rel)
    return DriftResult(imports_path=imports_path, missing=missing, changed=changed, ok=ok)


@dataclass(frozen=True)
class _McpServer:
    name: str
    raw_name: str
    transport: Literal["http", "stdio"]
    url: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    description: str | None = None
    timeout: int | None = None
    source_file: Path | None = None


def _discover_mcp_servers(
    source: ImportSource,
    source_dir: Path,
    warnings: list[str],
) -> list[_McpServer]:
    if source == "claude":
        return _discover_json_mcp_servers(_claude_mcp_config_candidates(source_dir), warnings)
    if source == "cursor":
        return _discover_json_mcp_servers(_cursor_mcp_config_candidates(source_dir), warnings)
    if source == "codex":
        return _discover_codex_mcp_servers(source_dir, warnings)
    if source == "pi":
        return _discover_json_mcp_servers(_generic_mcp_config_candidates(source_dir), warnings)
    raise AssertionError(source)


def _claude_mcp_config_candidates(source_dir: Path) -> list[Path]:
    candidates = [
        source_dir / ".claude.json",
        source_dir / "settings.json",
        source_dir / ".claude" / "settings.json",
        source_dir / ".mcp.json",
    ]
    if source_dir.name == ".claude":
        candidates.extend([source_dir.parent / ".claude.json", source_dir / "mcp.json"])
    return _dedupe_paths(candidates)


def _cursor_mcp_config_candidates(source_dir: Path) -> list[Path]:
    return _dedupe_paths(
        [
            source_dir / "mcp.json",
            source_dir / ".cursor" / "mcp.json",
            source_dir / "settings.json",
            source_dir / ".mcp.json",
        ]
    )


def _generic_mcp_config_candidates(source_dir: Path) -> list[Path]:
    return _dedupe_paths([source_dir / "mcp.json", source_dir / "settings.json"])


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def _discover_json_mcp_servers(paths: list[Path], warnings: list[str]) -> list[_McpServer]:
    servers: list[_McpServer] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Could not parse MCP config {path}: {exc}")
            continue
        block = raw.get("mcpServers") if isinstance(raw, dict) else None
        if not isinstance(block, dict):
            continue
        servers.extend(_normalize_mcp_mapping(block, path, seen, warnings))
    return servers


def _discover_codex_mcp_servers(source_dir: Path, warnings: list[str]) -> list[_McpServer]:
    config_path = source_dir / "config.toml"
    if not config_path.exists():
        return []
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        warnings.append(f"Could not parse Codex config {config_path}: {exc}")
        return []
    block = raw.get("mcp_servers")
    if not isinstance(block, dict):
        return []
    return _normalize_mcp_mapping(block, config_path, set(), warnings)


def _normalize_mcp_mapping(
    block: Mapping[str, object],
    source_file: Path,
    seen: set[str],
    warnings: list[str],
) -> list[_McpServer]:
    servers: list[_McpServer] = []
    for raw_name, raw_value in sorted(block.items()):
        if not isinstance(raw_value, dict):
            warnings.append(f"Skipping MCP server {raw_name!r} in {source_file}: not a mapping.")
            continue
        name = _safe_name(raw_name)
        if name in seen:
            warnings.append(
                f"Skipping MCP server {raw_name!r} in {source_file}: imported name {name!r} "
                "already exists."
            )
            continue
        normalized = _normalize_one_mcp_server(name, raw_name, raw_value, source_file, warnings)
        if normalized is None:
            continue
        seen.add(name)
        servers.append(normalized)
    return servers


def _normalize_one_mcp_server(
    name: str,
    raw_name: str,
    raw: Mapping[str, object],
    source_file: Path,
    warnings: list[str],
) -> _McpServer | None:
    url = _string_or_none(raw.get("url")) or _string_or_none(raw.get("serverUrl"))
    command = _string_or_none(raw.get("command"))
    raw_type = _string_or_none(raw.get("type")) or _string_or_none(raw.get("transport"))
    if raw_type in {"sse", "http", "streamable-http"} and url is not None:
        transport: Literal["http", "stdio"] = "http"
        if raw_type == "streamable-http":
            warnings.append(
                f"MCP server {raw_name!r} in {source_file} declares streamable-http; "
                "importing as Omnigent http transport."
            )
        if command is not None:
            warnings.append(
                f"MCP server {raw_name!r} in {source_file} declares both url and command; "
                "importing url as Omnigent http transport and dropping command."
            )
    elif command is not None:
        transport = "stdio"
        if url is not None:
            warnings.append(
                f"MCP server {raw_name!r} in {source_file} declares both command and url; "
                "importing command as Omnigent stdio transport and dropping url."
            )
    elif url is not None:
        transport = "http"
    else:
        warnings.append(
            f"Skipping MCP server {raw_name!r} in {source_file}: no supported command or url."
        )
        return None

    supported = {
        "args",
        "command",
        "description",
        "disabled",
        "env",
        "headers",
        "serverUrl",
        "timeout",
        "transport",
        "type",
        "url",
    }
    unsupported = sorted(str(k) for k in raw if str(k) not in supported)
    if unsupported:
        warnings.append(
            f"MCP server {raw_name!r} in {source_file} has unsupported field(s) "
            f"{unsupported}; they were not imported."
        )
    if raw.get("disabled") is True:
        warnings.append(f"Skipping disabled MCP server {raw_name!r} in {source_file}.")
        return None
    if transport == "http" and raw.get("env") is not None:
        warnings.append(
            f"MCP server {raw_name!r} in {source_file} has env on an HTTP transport; "
            "Omnigent MCP YAML only supports env for stdio, so env was not imported."
        )
    if transport == "stdio" and raw.get("headers") is not None:
        warnings.append(
            f"MCP server {raw_name!r} in {source_file} has headers on a stdio transport; "
            "Omnigent MCP YAML only supports headers for http, so headers were not imported."
        )

    return _McpServer(
        name=name,
        raw_name=raw_name,
        transport=transport,
        url=url if transport == "http" else None,
        command=command if transport == "stdio" else None,
        args=_string_list(raw.get("args"), raw_name, source_file, warnings),
        env=_string_dict(raw.get("env"), raw_name, source_file, "env", warnings),
        headers=_string_dict(raw.get("headers"), raw_name, source_file, "headers", warnings),
        description=_string_or_none(raw.get("description")),
        timeout=_int_or_none(raw.get("timeout"), raw_name, source_file, warnings),
        source_file=source_file,
    )


def _render_mcp_yaml(
    server: _McpServer,
    env_example: dict[str, str],
    *,
    force: bool,
    warnings: list[str],
) -> str:
    raw: dict[str, object] = {
        "name": server.name,
        "transport": server.transport,
    }
    if server.description:
        raw["description"] = server.description
    if server.timeout is not None:
        raw["timeout"] = server.timeout
    if server.transport == "http":
        if server.url is None:
            raw["url"] = None
        else:
            raw["url"] = _externalize_url(
                server.name,
                server.url,
                env_example,
                warnings=warnings,
            )
        if server.headers:
            raw["headers"] = _externalize_secret_mapping(
                server.name,
                server.headers,
                env_example,
                force=force,
                warnings=warnings,
            )
    else:
        raw["command"] = server.command
        if server.args:
            raw["args"] = _externalize_args(server, env_example, warnings=warnings)
        if server.env:
            raw["env"] = _externalize_secret_mapping(
                server.name,
                server.env,
                env_example,
                force=force,
                warnings=warnings,
            )
    return yaml.safe_dump(raw, sort_keys=False)


def _externalize_secret_mapping(
    server_name: str,
    values: Mapping[str, str],
    env_example: dict[str, str],
    *,
    force: bool,
    warnings: list[str],
) -> dict[str, str]:
    rendered: dict[str, str] = {}
    for key, value in values.items():
        env_refs = _embedded_env_refs(value)
        if env_refs:
            for ref in env_refs:
                env_example.setdefault(ref, "")
            rendered[key] = _normalize_env_reference_value(value)
            continue
        if not value:
            rendered[key] = value
            continue
        env_name = f"MCP_{_env_name(server_name)}_{_env_name(key)}"
        if force and not (_SECRET_KEY_RE.search(key) or _SECRET_VALUE_RE.match(value)):
            rendered[key] = value
            continue
        env_example.setdefault(env_name, "")
        rendered[key] = _replace_literal_with_ref(key, value, env_name)
        if force:
            warnings.append(
                f"Externalized secret-looking value for {server_name}.{key}; "
                "--force does not copy values that look like credentials."
            )
    return rendered


def _externalize_url(
    server_name: str,
    url: str,
    env_example: dict[str, str],
    *,
    warnings: list[str],
) -> str:
    split = urlsplit(url)
    if split.username or split.password or _secret_fragment(split.fragment):
        env_name = f"MCP_{_env_name(server_name)}_URL"
        env_example.setdefault(env_name, "")
        warnings.append(
            f"MCP server {server_name!r} has a secret-looking URL; externalized "
            f"the whole URL to ${{{env_name}}}."
        )
        return f"${{{env_name}}}"

    query = parse_qsl(split.query, keep_blank_values=True)
    if not query:
        return url
    changed = False
    rendered_query: list[tuple[str, str]] = []
    for key, value in query:
        env_refs = _embedded_env_refs(value)
        if env_refs:
            for ref in env_refs:
                env_example.setdefault(ref, "")
            rendered_query.append((key, _normalize_env_reference_value(value)))
            continue
        if _SECRET_KEY_RE.search(key) or _SECRET_VALUE_RE.match(value):
            env_name = f"MCP_{_env_name(server_name)}_{_env_name(key)}"
            env_example.setdefault(env_name, "")
            rendered_query.append((key, f"${{{env_name}}}"))
            changed = True
            continue
        rendered_query.append((key, value))
    if not changed:
        return url
    warnings.append(
        f"MCP server {server_name!r} has secret-looking URL query parameters; "
        "externalized them to environment variables."
    )
    return urlunsplit(
        (
            split.scheme,
            split.netloc,
            split.path,
            urlencode(rendered_query, doseq=True, safe="${}"),
            split.fragment,
        )
    )


def _secret_fragment(fragment: str) -> bool:
    return bool(fragment and _SECRET_VALUE_RE.match(fragment))


def _externalize_args(
    server: _McpServer,
    env_example: dict[str, str],
    *,
    warnings: list[str],
) -> list[str]:
    rendered: list[str] = []
    for idx, arg in enumerate(server.args):
        env_refs = _embedded_env_refs(arg)
        if env_refs:
            for ref in env_refs:
                env_example.setdefault(ref, "")
            rendered.append(_normalize_env_reference_value(arg))
            continue
        split = _split_arg_assignment(arg)
        if split is not None and (
            _SECRET_KEY_RE.search(split[0]) or _SECRET_VALUE_RE.match(split[1])
        ):
            env_name = f"MCP_{_env_name(server.name)}_ARG_{idx + 1}"
            env_example.setdefault(env_name, "")
            rendered.append(f"{split[0]}{split[2]}${{{env_name}}}")
            warnings.append(
                f"MCP server {server.raw_name!r} has a secret-looking literal in args; "
                f"externalized it to ${{{env_name}}}."
            )
            continue
        if _SECRET_VALUE_RE.match(arg):
            env_name = f"MCP_{_env_name(server.name)}_ARG_{idx + 1}"
            env_example.setdefault(env_name, "")
            rendered.append(f"${{{env_name}}}")
            warnings.append(
                f"MCP server {server.raw_name!r} has a secret-looking literal in args; "
                f"externalized it to ${{{env_name}}}."
            )
            continue
        rendered.append(arg)
    return rendered


def _split_arg_assignment(arg: str) -> tuple[str, str, str] | None:
    for sep in ("=", ":"):
        if sep in arg:
            left, right = arg.split(sep, 1)
            if left and right:
                return left, right, sep
    return None


def _replace_literal_with_ref(key: str, value: str, env_name: str) -> str:
    if key.lower() == "authorization" and value.lower().startswith("bearer "):
        return f"Bearer ${{{env_name}}}"
    return f"${{{env_name}}}"


def _discover_skill_dirs(source: ImportSource, source_dir: Path) -> list[Path]:
    candidates = [source_dir / "skills"]
    if source == "claude":
        candidates.extend([source_dir / ".claude" / "skills"])
        if source_dir.name == ".claude":
            candidates.append(source_dir / "skills")
    if source == "cursor":
        candidates.append(source_dir / ".cursor" / "skills")
    skills: dict[str, Path] = {}
    for root in _dedupe_paths(candidates):
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "SKILL.md").is_file():
                skills.setdefault(_safe_name(child.name), child)
    return list(skills.values())


def _scaffold_artifacts(
    source: ImportSource,
    into: Path,
    on_conflict: ConflictMode,
    prompt_resolver: PromptResolver | None,
) -> list[PlannedArtifact]:
    name = _safe_name(into.name or "imported-agent")
    harness = {
        "claude": "claude-native",
        "codex": "codex-native",
        "cursor": "cursor-native",
        "pi": "pi",
    }[source]
    config = yaml.safe_dump(
        {
            "spec_version": 1,
            "name": name,
            "description": f"Imported {source} harness assets.",
            "executor": {"type": "omnigent", "config": {"harness": harness}},
            "instructions": "AGENTS.md",
        },
        sort_keys=False,
    )
    prompt = (
        f"You are {name}, an Omnigent agent with imported {source} harness assets.\n\n"
        "Use the bundled MCP servers and skills when they are relevant to the task.\n"
    )
    return [
        _planned_file(
            into,
            "config.yaml",
            config,
            kind="scaffold",
            on_conflict=on_conflict,
            prompt_resolver=prompt_resolver,
        ),
        _planned_file(
            into,
            "AGENTS.md",
            prompt,
            kind="scaffold",
            on_conflict=on_conflict,
            prompt_resolver=prompt_resolver,
        ),
    ]


def _planned_file(
    into: Path,
    rel: str,
    content: str,
    *,
    kind: ImportKind,
    on_conflict: ConflictMode,
    prompt_resolver: PromptResolver | None,
) -> PlannedArtifact:
    target = into / rel
    if target.exists():
        if target.is_file() and target.read_text(encoding="utf-8") == content:
            return PlannedArtifact(
                kind=kind,
                relative_path=rel,
                action="unchanged",
                content=content,
            )
        mode = _resolve_conflict(on_conflict, rel, target, prompt_resolver)
        if mode == "skip":
            return PlannedArtifact(kind=kind, relative_path=rel, action="skip")
        if mode == "fail":
            raise OmnigentError(
                f"Import target already exists: {target}",
                code=ErrorCode.INVALID_INPUT,
            )
        return PlannedArtifact(kind=kind, relative_path=rel, action="update", content=content)
    return PlannedArtifact(kind=kind, relative_path=rel, action="create", content=content)


def _planned_dir(
    into: Path,
    rel: str,
    source_dir: Path,
    *,
    kind: ImportKind,
    on_conflict: ConflictMode,
    prompt_resolver: PromptResolver | None,
) -> PlannedArtifact:
    target = into / rel
    if target.exists():
        if target.is_dir() and _hash_path(target) == _hash_path(source_dir):
            return PlannedArtifact(
                kind=kind,
                relative_path=rel,
                action="unchanged",
                source_dir=source_dir,
            )
        mode = _resolve_conflict(on_conflict, rel, target, prompt_resolver)
        if mode == "skip":
            return PlannedArtifact(kind=kind, relative_path=rel, action="skip")
        if mode == "fail":
            raise OmnigentError(
                f"Import target already exists: {target}",
                code=ErrorCode.INVALID_INPUT,
            )
        return PlannedArtifact(
            kind=kind,
            relative_path=rel,
            action="update",
            source_dir=source_dir,
        )
    return PlannedArtifact(kind=kind, relative_path=rel, action="create", source_dir=source_dir)


def _resolve_conflict(
    mode: ConflictMode,
    rel: str,
    target: Path,
    prompt_resolver: PromptResolver | None,
) -> ConflictMode:
    if mode != "prompt":
        return mode
    if prompt_resolver is None:
        return "fail"
    selected = prompt_resolver(rel, target)
    if selected == "prompt":
        return "fail"
    return selected


def _validate_planned_bundle(plan: ImportPlan) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "bundle"
        if plan.into.exists():
            shutil.copytree(plan.into, tmp, dirs_exist_ok=True)
        else:
            tmp.mkdir(parents=True)
        for artifact in plan.artifacts:
            if artifact.action not in {"create", "update"}:
                continue
            target = tmp / artifact.relative_path
            if artifact.source_dir is not None:
                if target.exists():
                    _remove_path(target)
                shutil.copytree(artifact.source_dir, target)
            elif artifact.content is not None:
                if target.exists() and target.is_dir():
                    _remove_path(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(artifact.content, encoding="utf-8")
        load_spec(tmp, expand_env=False)


def _render_provenance(
    *,
    into: Path,
    source: ImportSource,
    source_dir: Path,
    artifacts: list[PlannedArtifact],
    warnings: list[str],
) -> str:
    changed = [a for a in artifacts if a.action in {"create", "update", "unchanged"}]
    hashes: dict[str, str] = {}
    for artifact in changed:
        if artifact.kind == "provenance":
            continue
        if artifact.source_dir is not None:
            hashes[artifact.relative_path] = _hash_path(artifact.source_dir)
        elif artifact.content is not None:
            hashes[artifact.relative_path] = _hash_text(artifact.content)

    entry = {
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "source_path": str(source_dir),
        "artifacts": hashes,
        "warnings": warnings,
    }
    imports: list[object] = []
    existing = into / ".omnigent" / "imports.yaml"
    if existing.exists():
        raw = yaml.safe_load(existing.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict) and isinstance(raw.get("imports"), list):
            imports = list(raw["imports"])
    imports.append(entry)
    return yaml.safe_dump({"imports": imports}, sort_keys=False)


def _merge_env_example(path: Path, new_vars: Mapping[str, str]) -> str:
    existing_lines: list[str] = []
    existing_keys: set[str] = set()
    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()
        for line in existing_lines:
            if "=" in line and not line.lstrip().startswith("#"):
                existing_keys.add(line.split("=", 1)[0])
    lines = list(existing_lines)
    if lines and lines[-1] != "":
        lines.append("")
    for key in sorted(new_vars):
        if key not in existing_keys:
            lines.append(f"{key}=")
    return "\n".join(lines).rstrip() + "\n"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _remove_path(path: Path) -> None:
    """Remove an existing file, symlink, or directory before overwrite."""

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _safe_name(value: str) -> str:
    name = _SAFE_NAME_RE.sub("_", value.strip()).strip("_")
    return name or "imported"


def _env_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper() or "VALUE"


def _embedded_env_refs(value: str) -> list[str]:
    return [m.group(1) or m.group(2) for m in _EMBEDDED_ENV_REF_RE.finditer(value)]


def _normalize_env_reference_value(value: str) -> str:
    match = _ENV_REF_RE.match(value)
    if match:
        return f"${{{match.group(1)}}}"
    return value


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _string_list(
    value: object,
    name: str,
    source_file: Path,
    warnings: list[str],
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        warnings.append(f"MCP server {name!r} in {source_file} has non-list args; skipped args.")
        return []
    return [str(v) for v in value]


def _string_dict(
    value: object,
    name: str,
    source_file: Path,
    field_name: str,
    warnings: list[str],
) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        warnings.append(
            f"MCP server {name!r} in {source_file} has non-mapping {field_name}; "
            f"skipped {field_name}."
        )
        return {}
    return {str(k): str(v) for k, v in value.items()}


def _int_or_none(
    value: object,
    name: str,
    source_file: Path,
    warnings: list[str],
) -> int | None:
    if value is None:
        return None
    if not isinstance(value, str | int | float):
        warnings.append(
            f"MCP server {name!r} in {source_file} has invalid timeout; skipped timeout."
        )
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        warnings.append(
            f"MCP server {name!r} in {source_file} has invalid timeout; skipped timeout."
        )
        return None


def default_conflict_mode() -> ConflictMode:
    """Default conflict behavior: prompt interactively, fail in scripts."""

    return "prompt" if sys.stdin.isatty() else "fail"


def env_bool(name: str) -> bool:
    """Return whether an environment variable is truthy."""

    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}
