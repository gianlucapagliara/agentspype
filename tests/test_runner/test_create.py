"""Tests for the agent scaffolding utility (copy_template)."""

from __future__ import annotations

from pathlib import Path

from agentspype.runner.create import _to_camel_case, _to_snake_case, copy_template

# ---------------------------------------------------------------------------
# Helper conversion tests
# ---------------------------------------------------------------------------


class TestCaseConversions:
    def test_to_snake_case_from_camel(self) -> None:
        assert _to_snake_case("MyAgent") == "my_agent"

    def test_to_snake_case_already_snake(self) -> None:
        assert _to_snake_case("my_agent") == "my_agent"

    def test_to_snake_case_single_word(self) -> None:
        assert _to_snake_case("Agent") == "agent"

    def test_to_camel_case_from_snake(self) -> None:
        assert _to_camel_case("my_agent") == "MyAgent"

    def test_to_camel_case_already_camel(self) -> None:
        assert _to_camel_case("MyAgent") == "MyAgent"

    def test_to_camel_case_single_word(self) -> None:
        assert _to_camel_case("agent") == "Agent"


# ---------------------------------------------------------------------------
# copy_template tests
# ---------------------------------------------------------------------------


class TestCopyTemplate:
    def test_creates_directory(self, tmp_path: Path) -> None:
        output = tmp_path / "my_new_agent"
        result = copy_template("MyNew", output)
        assert result == output
        assert output.is_dir()

    def test_creates_expected_files(self, tmp_path: Path) -> None:
        output = tmp_path / "my_agent"
        copy_template("MyTest", output)

        expected_files = [
            "__init__.py",
            "agent.py",
            "configuration.py",
            "listening.py",
            "publishing.py",
            "state_machine.py",
            "status.py",
        ]
        for filename in expected_files:
            assert (output / filename).exists(), f"Missing: {filename}"

    def test_renames_template_class_names(self, tmp_path: Path) -> None:
        output = tmp_path / "cool_agent"
        copy_template("Cool", output)

        agent_py = (output / "agent.py").read_text()
        assert "CoolAgent" in agent_py
        assert "TemplateAgent" not in agent_py
        assert "Template" not in agent_py

    def test_renames_import_paths(self, tmp_path: Path) -> None:
        output = tmp_path / "agents" / "my_agent"
        copy_template("MyAgent", output)

        agent_py = (output / "agent.py").read_text()
        # The import path should use the output path parts, not agentspype.template
        assert "agentspype.template." not in agent_py

    def test_does_not_overwrite_existing_files(self, tmp_path: Path) -> None:
        output = tmp_path / "my_agent"
        output.mkdir(parents=True)
        sentinel = output / "agent.py"
        sentinel.write_text("# custom content\n")

        copy_template("MyAgent", output)

        # The existing file should not be overwritten
        assert sentinel.read_text() == "# custom content\n"

    def test_camel_case_from_snake_input(self, tmp_path: Path) -> None:
        output = tmp_path / "my_agent"
        copy_template("my_fancy_agent", output)

        agent_py = (output / "agent.py").read_text()
        # "my_fancy_agent" → CamelCase "MyFancyAgent" → strip "Agent" → prefix "MyFancy"
        # so TemplateAgent becomes MyFancyAgent (not MyFancyAgentAgent)
        assert "MyFancyAgent" in agent_py
        assert "MyFancyAgentAgent" not in agent_py

    def test_absolute_path_import_has_no_slashes(self, tmp_path: Path) -> None:
        """An absolute output_dir should produce an import path without '/' or root components."""
        output = tmp_path / "agents" / "my_agent"
        copy_template("Cool", output)

        agent_py = (output / "agent.py").read_text()
        # Extract the import path that replaced "agentspype.template."
        # It should be a dotted path, never containing "/" or bare root parts
        for line in agent_py.splitlines():
            if "import" in line or "from" in line:
                assert "/" not in line, f"Import line contains '/': {line}"
