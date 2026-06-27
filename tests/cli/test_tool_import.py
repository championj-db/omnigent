"""Tests for ``omnigent tool import``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from omnigent.tool_import import (
    apply_import_plan,
    check_import_drift,
    plan_import,
)
from omnigent.cli import _resolve_bundle_env_vars, cli
from omnigent.errors import OmnigentError
from omnigent.spec import load as load_spec


def test_claude_mcp_import_externalizes_secrets_and_validates(tmp_path: Path) -> None:
    """Claude MCP servers are materialized as normal Omnigent MCP YAML."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (tmp_path / ".claude.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "github": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-github"],
                        "env": {"GITHUB_TOKEN": "ghp_abcdefghijklmnopqrstuvwxyz"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    into = tmp_path / "my-agent"

    plan = plan_import(
        source="claude",
        source_dir=claude_dir,
        into=into,
        import_mcp=True,
        import_skills=False,
        dry_run=False,
        on_conflict="fail",
    )
    apply_import_plan(plan)

    mcp = yaml.safe_load((into / "tools" / "mcp" / "github.yaml").read_text())
    assert mcp == {
        "name": "github",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {"GITHUB_TOKEN": "${MCP_GITHUB_GITHUB_TOKEN}"},
    }
    assert (into / ".env.example").read_text(encoding="utf-8") == (
        "MCP_GITHUB_GITHUB_TOKEN=\n"
    )
    assert load_spec(into, expand_env=False).mcp_servers[0].name == "github"
    assert check_import_drift(into).clean is True


def test_codex_mcp_import_reads_config_toml(tmp_path: Path) -> None:
    """Codex ``[mcp_servers.<name>]`` tables import as stdio MCP YAML."""
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "config.toml").write_text(
        """
[mcp_servers.filesystem]
command = "node"
args = ["server.js"]
env = { ROOT = "$PROJECT_ROOT" }
""".strip(),
        encoding="utf-8",
    )
    into = tmp_path / "agent"

    plan = plan_import(
        source="codex",
        source_dir=codex_dir,
        into=into,
        import_mcp=True,
        import_skills=False,
        dry_run=False,
        on_conflict="fail",
    )
    apply_import_plan(plan)

    mcp = yaml.safe_load((into / "tools" / "mcp" / "filesystem.yaml").read_text())
    assert mcp["command"] == "node"
    assert mcp["args"] == ["server.js"]
    assert mcp["env"] == {"ROOT": "${PROJECT_ROOT}"}
    assert (into / ".env.example").read_text(encoding="utf-8") == "PROJECT_ROOT=\n"


def test_force_preserves_non_secret_literals_without_env_example_noise(
    tmp_path: Path,
) -> None:
    """``--force`` keeps non-secret values literal and externalizes secrets."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (tmp_path / ".claude.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "headers": {
                        "url": "https://mcp.example/sse",
                        "headers": {
                            "Authorization": "Bearer secret-token-12345678901234567890",
                            "X-Workspace": "dev",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    into = tmp_path / "agent"

    plan = plan_import(
        source="claude",
        source_dir=claude_dir,
        into=into,
        import_mcp=True,
        import_skills=False,
        dry_run=False,
        on_conflict="fail",
        force=True,
    )
    apply_import_plan(plan)

    mcp = yaml.safe_load((into / "tools" / "mcp" / "headers.yaml").read_text())
    assert mcp["headers"] == {
        "Authorization": "Bearer ${MCP_HEADERS_AUTHORIZATION}",
        "X-Workspace": "dev",
    }
    assert (into / ".env.example").read_text(encoding="utf-8") == (
        "MCP_HEADERS_AUTHORIZATION=\n"
    )


def test_http_url_query_values_are_imported_as_is(tmp_path: Path) -> None:
    """HTTP MCP URLs are copied without changing query parameters."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (tmp_path / ".claude.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "remote": {
                        "url": "https://mcp.example/sse?api_key=sk-abcdefghijklmnop&team=dev"
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    into = tmp_path / "agent"

    plan = plan_import(
        source="claude",
        source_dir=claude_dir,
        into=into,
        import_mcp=True,
        import_skills=False,
        dry_run=False,
        on_conflict="fail",
    )
    apply_import_plan(plan)

    mcp_text = (into / "tools" / "mcp" / "remote.yaml").read_text(encoding="utf-8")
    mcp = yaml.safe_load(mcp_text)
    assert mcp["url"] == "https://mcp.example/sse?api_key=sk-abcdefghijklmnop&team=dev"
    assert not (into / ".env.example").exists()
    assert not any("URL query parameters" in warning for warning in plan.warnings)


def test_http_url_hosted_path_ids_are_not_externalized(tmp_path: Path) -> None:
    """Opaque hosted MCP path IDs are not treated as credentials."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    hosted_url = (
        "https://hosted.mcp.example/servers/"
        "0123456789abcdef0123456789abcdef/sse?team=dev"
    )
    (tmp_path / ".claude.json").write_text(
        json.dumps({"mcpServers": {"hosted": {"url": hosted_url}}}),
        encoding="utf-8",
    )
    into = tmp_path / "agent"

    plan = plan_import(
        source="claude",
        source_dir=claude_dir,
        into=into,
        import_mcp=True,
        import_skills=False,
        dry_run=False,
        on_conflict="fail",
    )
    apply_import_plan(plan)

    mcp = yaml.safe_load(
        (into / "tools" / "mcp" / "hosted.yaml").read_text(encoding="utf-8")
    )
    assert mcp["url"] == hosted_url
    assert not (into / ".env.example").exists()
    assert not any("secret-looking URL" in warning for warning in plan.warnings)


def test_stdio_arg_secrets_are_externalized(tmp_path: Path) -> None:
    """Secret-looking stdio args are externalized instead of copied."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (tmp_path / ".claude.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "local": {
                        "command": "node",
                        "args": ["server.js", "--token=sk-abcdefghijklmnop"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    into = tmp_path / "agent"

    plan = plan_import(
        source="claude",
        source_dir=claude_dir,
        into=into,
        import_mcp=True,
        import_skills=False,
        dry_run=False,
        on_conflict="fail",
    )
    apply_import_plan(plan)

    mcp_text = (into / "tools" / "mcp" / "local.yaml").read_text(encoding="utf-8")
    assert "sk-abcdefghijklmnop" not in mcp_text
    mcp = yaml.safe_load(mcp_text)
    assert mcp["args"] == ["server.js", "--token=${MCP_LOCAL_ARG_2}"]
    assert (into / ".env.example").read_text(encoding="utf-8") == "MCP_LOCAL_ARG_2=\n"


def test_bundle_env_resolution_expands_mcp_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Client-side bundling resolves MCP URL env refs for uploaded bundles."""
    monkeypatch.setenv("MCP_REMOTE_API_KEY", "sk-from-env")
    agent = tmp_path / "agent"
    mcp_dir = agent / "tools" / "mcp"
    mcp_dir.mkdir(parents=True)
    (agent / "config.yaml").write_text(
        yaml.dump(
            {
                "spec_version": 1,
                "name": "agent",
                "executor": {"type": "omnigent", "config": {"harness": "claude-sdk"}},
            }
        ),
        encoding="utf-8",
    )
    (mcp_dir / "remote.yaml").write_text(
        yaml.dump(
            {
                "name": "remote",
                "transport": "http",
                "url": "https://mcp.example/sse?api_key=${MCP_REMOTE_API_KEY}",
            }
        ),
        encoding="utf-8",
    )

    resolved = _resolve_bundle_env_vars(agent)

    remote = yaml.safe_load(resolved["tools/mcp/remote.yaml"])
    assert remote["url"] == "https://mcp.example/sse?api_key=sk-from-env"


def test_lossy_wrong_transport_fields_warn(tmp_path: Path) -> None:
    """Source fields unsupported by Omnigent's target transport are warned."""
    cursor_dir = tmp_path / ".cursor"
    cursor_dir.mkdir()
    (cursor_dir / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "remote": {
                        "url": "https://mcp.example/sse",
                        "env": {"TOKEN": "$TOKEN"},
                    },
                    "local": {
                        "command": "node",
                        "headers": {"Authorization": "$TOKEN"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    plan = plan_import(
        source="cursor",
        source_dir=cursor_dir,
        into=tmp_path / "agent",
        import_mcp=True,
        import_skills=False,
        dry_run=True,
        on_conflict="fail",
    )

    assert any("env on an HTTP transport" in warning for warning in plan.warnings)
    assert any("headers on a stdio transport" in warning for warning in plan.warnings)


def test_import_skills_copies_skill_directories_without_config_skills_field(
    tmp_path: Path,
) -> None:
    """Skills import copies bundled skills and preserves ``skills:`` semantics."""
    cursor_dir = tmp_path / ".cursor"
    skill_dir = cursor_dir / "skills" / "triage"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: triage\ndescription: Triage issues.\n---\n\nDo triage.\n",
        encoding="utf-8",
    )
    into = tmp_path / "agent"

    plan = plan_import(
        source="cursor",
        source_dir=cursor_dir,
        into=into,
        import_mcp=False,
        import_skills=True,
        dry_run=False,
        on_conflict="fail",
    )
    apply_import_plan(plan)

    assert (into / "skills" / "triage" / "SKILL.md").is_file()
    config = yaml.safe_load((into / "config.yaml").read_text())
    assert "skills" not in config
    assert config["executor"]["config"]["harness"] == "cursor"


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    """Dry-run validates the would-be bundle without touching disk."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (tmp_path / ".claude.json").write_text(
        json.dumps({"mcpServers": {"search": {"url": "https://mcp.example/sse"}}}),
        encoding="utf-8",
    )
    into = tmp_path / "agent"

    plan = plan_import(
        source="claude",
        source_dir=claude_dir,
        into=into,
        import_mcp=True,
        import_skills=False,
        dry_run=True,
        on_conflict="fail",
    )
    apply_import_plan(plan)

    assert plan.changed_artifacts
    assert not into.exists()


def test_check_reports_drift(tmp_path: Path) -> None:
    """``--check`` detects local edits to imported artifacts."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (tmp_path / ".claude.json").write_text(
        json.dumps({"mcpServers": {"search": {"url": "https://mcp.example/sse"}}}),
        encoding="utf-8",
    )
    into = tmp_path / "agent"
    plan = plan_import(
        source="claude",
        source_dir=claude_dir,
        into=into,
        import_mcp=True,
        import_skills=False,
        dry_run=False,
        on_conflict="fail",
    )
    apply_import_plan(plan)

    (into / "tools" / "mcp" / "search.yaml").write_text(
        "name: search\ntransport: http\nurl: https://changed.example/sse\n",
        encoding="utf-8",
    )

    result = check_import_drift(into)
    assert result.clean is False
    assert result.changed == ["tools/mcp/search.yaml"]


def test_cli_tool_import_claude_dry_run(tmp_path: Path) -> None:
    """Click command exposes the requested ``tool import`` shape."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (tmp_path / ".claude.json").write_text(
        json.dumps({"mcpServers": {"search": {"url": "https://mcp.example/sse"}}}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        [
            "tool",
            "import",
            "claude",
            "--from",
            str(claude_dir),
            "--into",
            str(tmp_path / "agent"),
            "--mcp",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Would write tools/mcp/search.yaml" in result.output
    assert "Dry run complete" in result.output


def test_tool_import_replaces_removed_agent_command() -> None:
    """The branch-added import command lives under ``tool``, not ``agent``."""
    assert "tool" in cli.commands
    assert "agent" not in cli.commands
    assert "import" in cli.commands["tool"].commands


def test_pi_requires_explicit_source_dir(tmp_path: Path) -> None:
    """Pi is a consumer unless the caller supplies an enumerable directory."""
    with pytest.raises(OmnigentError, match="Pi is not an import source"):
        plan_import(
            source="pi",
            source_dir=None,
            into=tmp_path / "agent",
            import_mcp=True,
            import_skills=False,
            dry_run=True,
            on_conflict="fail",
        )
