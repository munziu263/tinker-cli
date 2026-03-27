"""Tests for tinker CLI - written FIRST (red phase of TDD)."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def run_tinker(*args, cwd=None, stdin_text=None):
    """Run the tinker CLI as a subprocess and return (stdout, stderr, returncode)."""
    result = subprocess.run(
        [sys.executable, "-m", "tinker.cli", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        input=stdin_text,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")},
    )
    return result.stdout, result.stderr, result.returncode


@pytest.fixture
def project_dir(tmp_path):
    """Create a fake project directory with a .git marker."""
    (tmp_path / ".git").mkdir()
    return tmp_path


# === tinker --version ===

class TestVersion:
    def test_version_prints_version(self):
        stdout, stderr, rc = run_tinker("--version")
        assert rc == 0
        assert "0.1.0" in stdout


# === tinker --help ===

class TestHelp:
    def test_help_contains_description(self):
        stdout, stderr, rc = run_tinker("--help")
        assert rc == 0
        assert "tinker - Build interactive demo files" in stdout

    def test_help_contains_commands(self):
        stdout, _, _ = run_tinker("--help")
        assert "tinker init" in stdout
        assert "tinker cell" in stdout
        assert "tinker code" in stdout
        assert "tinker set-command" in stdout
        assert "tinker run" in stdout
        assert "tinker show" in stdout
        assert "tinker pop" in stdout
        assert "tinker list" in stdout

    def test_help_contains_workflow(self):
        stdout, _, _ = run_tinker("--help")
        assert "Workflow:" in stdout

    def test_help_contains_file_formats(self):
        stdout, _, _ = run_tinker("--help")
        assert "File formats:" in stdout

    def test_help_contains_languages(self):
        stdout, _, _ = run_tinker("--help")
        assert "Languages:" in stdout
        assert "python, c, zig, go, javascript, bash" in stdout

    def test_help_contains_stdin_section(self):
        stdout, _, _ = run_tinker("--help")
        assert "Stdin:" in stdout

    def test_help_contains_directory_structure(self):
        stdout, _, _ = run_tinker("--help")
        assert "Directory structure:" in stdout


# === tinker init ===

class TestInit:
    def test_init_python(self, project_dir):
        stdout, stderr, rc = run_tinker("init", "my-demo", "--lang", "python", cwd=project_dir)
        assert rc == 0
        assert ".tinker/my-demo/demo.py" in stdout

        demo_file = project_dir / ".tinker" / "my-demo" / "demo.py"
        assert demo_file.exists()
        content = demo_file.read_text()
        assert "# %% [markdown]" in content
        assert "# # my-demo" in content
        assert "# *Created by tinker*" in content

        toml_file = project_dir / ".tinker" / "my-demo" / "tinker.toml"
        assert toml_file.exists()
        toml_content = toml_file.read_text()
        assert 'name = "my-demo"' in toml_content
        assert 'lang = "python"' in toml_content
        assert 'command = ""' in toml_content

    def test_init_c(self, project_dir):
        stdout, _, rc = run_tinker("init", "bench", "--lang", "c", cwd=project_dir)
        assert rc == 0
        assert ".tinker/bench/demo.c" in stdout
        demo = project_dir / ".tinker" / "bench" / "demo.c"
        content = demo.read_text()
        assert "// bench" in content
        assert "// Created by tinker" in content

    def test_init_zig(self, project_dir):
        stdout, _, rc = run_tinker("init", "test-zig", "--lang", "zig", cwd=project_dir)
        assert rc == 0
        assert ".tinker/test-zig/demo.zig" in stdout

    def test_init_go(self, project_dir):
        stdout, _, rc = run_tinker("init", "test-go", "--lang", "go", cwd=project_dir)
        assert rc == 0
        assert ".tinker/test-go/demo.go" in stdout

    def test_init_javascript(self, project_dir):
        stdout, _, rc = run_tinker("init", "test-js", "--lang", "javascript", cwd=project_dir)
        assert rc == 0
        assert ".tinker/test-js/demo.js" in stdout

    def test_init_bash(self, project_dir):
        stdout, _, rc = run_tinker("init", "test-sh", "--lang", "bash", cwd=project_dir)
        assert rc == 0
        assert ".tinker/test-sh/demo.sh" in stdout
        demo = project_dir / ".tinker" / "test-sh" / "demo.sh"
        content = demo.read_text()
        assert "# test-sh" in content
        assert "# Created by tinker" in content

    def test_init_already_exists(self, project_dir):
        run_tinker("init", "dup", "--lang", "python", cwd=project_dir)
        _, stderr, rc = run_tinker("init", "dup", "--lang", "python", cwd=project_dir)
        assert rc == 1
        assert "already exists" in stderr

    def test_init_invalid_lang(self, project_dir):
        _, stderr, rc = run_tinker("init", "bad", "--lang", "rust", cwd=project_dir)
        assert rc == 1

    def test_init_rejects_path_traversal(self, project_dir):
        _, stderr, rc = run_tinker("init", "../../etc/passwd", "--lang", "python", cwd=project_dir)
        assert rc == 1
        assert "invalid" in stderr.lower()

    def test_init_rejects_slash(self, project_dir):
        _, stderr, rc = run_tinker("init", "foo/bar", "--lang", "python", cwd=project_dir)
        assert rc == 1
        assert "invalid" in stderr.lower()

    def test_init_rejects_backslash(self, project_dir):
        _, stderr, rc = run_tinker("init", "foo\\bar", "--lang", "python", cwd=project_dir)
        assert rc == 1
        assert "invalid" in stderr.lower()

    def test_init_finds_project_root_from_subdir(self, project_dir):
        subdir = project_dir / "src" / "foo"
        subdir.mkdir(parents=True)
        stdout, _, rc = run_tinker("init", "deep", "--lang", "python", cwd=subdir)
        assert rc == 0
        assert (project_dir / ".tinker" / "deep" / "demo.py").exists()


# === tinker cell ===

class TestCell:
    def test_cell_python(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        _, _, rc = run_tinker("cell", "demo", "This is commentary", cwd=project_dir)
        assert rc == 0
        content = (project_dir / ".tinker" / "demo" / "demo.py").read_text()
        assert "# %% [markdown]" in content
        assert "# This is commentary" in content

    def test_cell_c(self, project_dir):
        run_tinker("init", "demo", "--lang", "c", cwd=project_dir)
        _, _, rc = run_tinker("cell", "demo", "Test commentary", cwd=project_dir)
        assert rc == 0
        content = (project_dir / ".tinker" / "demo" / "demo.c").read_text()
        assert "// ---" in content
        assert "// Test commentary" in content

    def test_cell_bash(self, project_dir):
        run_tinker("init", "demo", "--lang", "bash", cwd=project_dir)
        _, _, rc = run_tinker("cell", "demo", "Bash commentary", cwd=project_dir)
        assert rc == 0
        content = (project_dir / ".tinker" / "demo" / "demo.sh").read_text()
        assert "# ---" in content
        assert "# Bash commentary" in content

    def test_cell_from_stdin(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        _, _, rc = run_tinker("cell", "demo", cwd=project_dir, stdin_text="From stdin")
        assert rc == 0
        content = (project_dir / ".tinker" / "demo" / "demo.py").read_text()
        assert "# From stdin" in content

    def test_cell_not_found(self, project_dir):
        _, stderr, rc = run_tinker("cell", "nope", "text", cwd=project_dir)
        assert rc == 1
        assert "not found" in stderr

    def test_cell_trailing_blank_line(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        run_tinker("cell", "demo", "Test", cwd=project_dir)
        content = (project_dir / ".tinker" / "demo" / "demo.py").read_text()
        assert content.endswith("\n\n")

    def test_cell_multiline(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        run_tinker("cell", "demo", "Line one\nLine two", cwd=project_dir)
        content = (project_dir / ".tinker" / "demo" / "demo.py").read_text()
        assert "# Line one\n# Line two" in content


# === tinker code ===

class TestCode:
    def test_code_python(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        _, _, rc = run_tinker("code", "demo", "x = 1\nprint(x)", cwd=project_dir)
        assert rc == 0
        content = (project_dir / ".tinker" / "demo" / "demo.py").read_text()
        assert "# %%" in content
        assert "x = 1\nprint(x)" in content

    def test_code_c(self, project_dir):
        run_tinker("init", "demo", "--lang", "c", cwd=project_dir)
        _, _, rc = run_tinker("code", "demo", '#include <stdio.h>\nint main() { return 0; }', cwd=project_dir)
        assert rc == 0
        content = (project_dir / ".tinker" / "demo" / "demo.c").read_text()
        assert '#include <stdio.h>' in content
        # For compiled languages, no cell delimiter
        # The content should NOT have "# %%" in it
        lines_with_delimiters = [l for l in content.split("\n") if l.strip() == "# %%" or l.strip() == "// %%"]
        assert len(lines_with_delimiters) == 0

    def test_code_from_stdin(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        _, _, rc = run_tinker("code", "demo", cwd=project_dir, stdin_text="y = 2\nprint(y)")
        assert rc == 0
        content = (project_dir / ".tinker" / "demo" / "demo.py").read_text()
        assert "y = 2\nprint(y)" in content

    def test_code_not_found(self, project_dir):
        _, stderr, rc = run_tinker("code", "nope", "x=1", cwd=project_dir)
        assert rc == 1
        assert "not found" in stderr

    def test_code_trailing_blank_line(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        run_tinker("code", "demo", "x = 1", cwd=project_dir)
        content = (project_dir / ".tinker" / "demo" / "demo.py").read_text()
        assert content.endswith("\n\n")


# === tinker set-command ===

class TestSetCommand:
    def test_set_command(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        stdout, _, rc = run_tinker("set-command", "demo", "python .tinker/demo/demo.py", cwd=project_dir)
        assert rc == 0
        toml = (project_dir / ".tinker" / "demo" / "tinker.toml").read_text()
        assert "PYTHONPATH=" in toml  # auto-prepended for python
        assert "PYTHONPATH=" in stdout

    def test_set_command_no_pythonpath_for_c(self, project_dir):
        run_tinker("init", "demo", "--lang", "c", cwd=project_dir)
        stdout, _, rc = run_tinker("set-command", "demo", "gcc demo.c -o demo && ./demo", cwd=project_dir)
        assert rc == 0
        assert "PYTHONPATH" not in stdout

    def test_set_command_pythonpath_already_present(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        cmd = "PYTHONPATH=/custom python demo.py"
        stdout, _, rc = run_tinker("set-command", "demo", cmd, cwd=project_dir)
        assert rc == 0
        # Should not double-add PYTHONPATH
        assert stdout.count("PYTHONPATH") == 1

    def test_set_command_prints_final_command(self, project_dir):
        run_tinker("init", "demo", "--lang", "c", cwd=project_dir)
        stdout, _, rc = run_tinker("set-command", "demo", "make run", cwd=project_dir)
        assert rc == 0
        assert "make run" in stdout


# === tinker set-repl ===

class TestSetRepl:
    def test_set_repl_writes_toml(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        stdout, _, rc = run_tinker("set-repl", "demo", ".venv/bin/ipython", cwd=project_dir)
        assert rc == 0
        assert ".venv/bin/ipython" in stdout
        toml_path = project_dir / ".tinker" / "demo" / "tinker.toml"
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        data = tomllib.loads(toml_path.read_text())
        assert data["repl"]["cmd"] == ".venv/bin/ipython"
        assert data["repl"]["startup"] == []

    def test_set_repl_with_startup(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        stdout, _, rc = run_tinker(
            "set-repl", "demo", ".venv/bin/ipython",
            "--startup", "%load_ext autoreload",
            "--startup", "%autoreload 2",
            cwd=project_dir,
        )
        assert rc == 0
        toml_path = project_dir / ".tinker" / "demo" / "tinker.toml"
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        data = tomllib.loads(toml_path.read_text())
        assert data["repl"]["cmd"] == ".venv/bin/ipython"
        assert data["repl"]["startup"] == ["%load_ext autoreload", "%autoreload 2"]

    def test_set_repl_nonexistent_demo(self, project_dir):
        _, stderr, rc = run_tinker("set-repl", "nope", ".venv/bin/ipython", cwd=project_dir)
        assert rc == 1
        assert "not found" in stderr


# === tinker run ===

class TestRun:
    def test_run_success(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        run_tinker("code", "demo", "print('hello from tinker')", cwd=project_dir)
        run_tinker("set-command", "demo", f"python .tinker/demo/demo.py", cwd=project_dir)
        stdout, _, rc = run_tinker("run", "demo", cwd=project_dir)
        assert rc == 0
        assert "hello from tinker" in stdout

    def test_run_no_command_set(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        _, stderr, rc = run_tinker("run", "demo", cwd=project_dir)
        assert rc == 1
        assert "no command set" in stderr

    def test_run_not_found(self, project_dir):
        _, stderr, rc = run_tinker("run", "nope", cwd=project_dir)
        assert rc == 1
        assert "not found" in stderr

    def test_run_failure_exit_code(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        run_tinker("code", "demo", "import sys; sys.exit(42)", cwd=project_dir)
        run_tinker("set-command", "demo", "python .tinker/demo/demo.py", cwd=project_dir)
        _, _, rc = run_tinker("run", "demo", cwd=project_dir)
        assert rc == 42


# === tinker show ===

class TestShow:
    def test_show(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        run_tinker("code", "demo", "x = 1", cwd=project_dir)
        run_tinker("set-command", "demo", "python demo.py", cwd=project_dir)
        stdout, _, rc = run_tinker("show", "demo", cwd=project_dir)
        assert rc == 0
        assert "x = 1" in stdout
        assert "--- tinker command ---" in stdout

    def test_show_not_found(self, project_dir):
        _, stderr, rc = run_tinker("show", "nope", cwd=project_dir)
        assert rc == 1
        assert "not found" in stderr


# === tinker pop ===

class TestPop:
    def test_pop_python(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        run_tinker("cell", "demo", "First addition", cwd=project_dir)
        run_tinker("code", "demo", "x = 1", cwd=project_dir)
        # Now pop should remove the last cell (the code cell)
        _, _, rc = run_tinker("pop", "demo", cwd=project_dir)
        assert rc == 0
        content = (project_dir / ".tinker" / "demo" / "demo.py").read_text()
        assert "x = 1" not in content
        # The first addition cell should still be there
        assert "First addition" in content

    def test_pop_only_header_errors(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        _, stderr, rc = run_tinker("pop", "demo", cwd=project_dir)
        assert rc == 1

    def test_pop_c(self, project_dir):
        run_tinker("init", "demo", "--lang", "c", cwd=project_dir)
        run_tinker("cell", "demo", "A comment", cwd=project_dir)
        run_tinker("code", "demo", "int main() { return 0; }", cwd=project_dir)
        _, _, rc = run_tinker("pop", "demo", cwd=project_dir)
        assert rc == 0
        content = (project_dir / ".tinker" / "demo" / "demo.c").read_text()
        assert "int main" not in content
        # The cell comment should still be there
        assert "A comment" in content

    def test_pop_not_found(self, project_dir):
        _, stderr, rc = run_tinker("pop", "nope", cwd=project_dir)
        assert rc == 1
        assert "not found" in stderr


# === tinker list ===

class TestList:
    def test_list_empty(self, project_dir):
        stdout, _, rc = run_tinker("list", cwd=project_dir)
        assert rc == 0

    def test_list_with_demos(self, project_dir):
        run_tinker("init", "demo1", "--lang", "python", cwd=project_dir)
        run_tinker("init", "demo2", "--lang", "c", cwd=project_dir)
        stdout, _, rc = run_tinker("list", cwd=project_dir)
        assert rc == 0
        assert "demo1" in stdout
        assert "python" in stdout
        assert "demo2" in stdout
        assert "c" in stdout

    def test_list_shows_command_status(self, project_dir):
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        stdout, _, _ = run_tinker("list", cwd=project_dir)
        assert "no command set" in stdout

        run_tinker("set-command", "demo", "python demo.py", cwd=project_dir)
        stdout, _, _ = run_tinker("list", cwd=project_dir)
        # The command includes auto-prepended PYTHONPATH, check it's not "no command set"
        assert "no command set" not in stdout
        assert "python demo.py" in stdout or "PYTHONPATH" in stdout


# === Project root finding ===

class TestProjectRoot:
    def test_finds_pyproject_toml(self, tmp_path):
        (tmp_path / "pyproject.toml").touch()
        subdir = tmp_path / "src"
        subdir.mkdir()
        stdout, _, rc = run_tinker("init", "demo", "--lang", "python", cwd=subdir)
        assert rc == 0
        assert (tmp_path / ".tinker" / "demo" / "demo.py").exists()

    def test_finds_package_json(self, tmp_path):
        (tmp_path / "package.json").touch()
        stdout, _, rc = run_tinker("init", "demo", "--lang", "javascript", cwd=tmp_path)
        assert rc == 0
        assert (tmp_path / ".tinker" / "demo" / "demo.js").exists()

    def test_falls_back_to_cwd(self, tmp_path):
        # No project markers at all
        stdout, _, rc = run_tinker("init", "demo", "--lang", "python", cwd=tmp_path)
        assert rc == 0
        assert (tmp_path / ".tinker" / "demo" / "demo.py").exists()


# === TOML reading/writing ===

class TestTomlRoundtrip:
    def test_toml_manifest_readable(self, project_dir):
        """The tinker.toml should be valid TOML that can be parsed."""
        run_tinker("init", "demo", "--lang", "python", cwd=project_dir)
        toml_path = project_dir / ".tinker" / "demo" / "tinker.toml"
        content = toml_path.read_text()
        # Should be parseable
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        data = tomllib.loads(content)
        assert data["name"] == "demo"
        assert data["lang"] == "python"
        assert data["command"] == ""

    def test_toml_after_set_command(self, project_dir):
        run_tinker("init", "demo", "--lang", "c", cwd=project_dir)
        run_tinker("set-command", "demo", "gcc demo.c && ./a.out", cwd=project_dir)
        toml_path = project_dir / ".tinker" / "demo" / "tinker.toml"
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        data = tomllib.loads(toml_path.read_text())
        assert data["command"] == "gcc demo.c && ./a.out"
