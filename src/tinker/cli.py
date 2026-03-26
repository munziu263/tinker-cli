"""Tinker CLI - Build interactive demo files for exploring code changes."""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import tomli_w

from tinker import __version__

SUPPORTED_LANGS = ("python", "c", "zig", "go", "javascript", "bash")

LANG_EXT = {
    "python": "py",
    "c": "c",
    "zig": "zig",
    "go": "go",
    "javascript": "js",
    "bash": "sh",
}

PROJECT_MARKERS = (".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod")

HELP_TEXT = """\
tinker - Build interactive demo files for exploring code changes.

Tinker helps AI agents create executable demo files that users can open
in their editor and run interactively. For Python, it produces percent-
format scripts with # %% cell delimiters suitable for REPL-based
exploration. For compiled languages, it produces standalone source files
with a run command.

Usage:
  tinker init <name> --lang <lang>       Create a new demo
  tinker cell <name> [text]              Append commentary (text or stdin)
  tinker code <name> [code]              Append code (code or stdin)
  tinker set-command <name> <command>    Set the run command
  tinker run <name>                      Execute the demo
  tinker show <name>                     Print demo file and command
  tinker pop <name>                      Remove the last cell
  tinker list                            List all demos in this project
  tinker --version                       Print version
  tinker --help                          Show this help

Languages:
  python, c, zig, go, javascript, bash

File formats:
  Python demos use the percent format (# %% cell delimiters) for REPL-
  based exploration. Each cell can be sent individually to an IPython
  session. Commentary cells use # %% [markdown].

  Compiled language demos are standalone source files. Commentary is
  written as block comments. The entire file is compiled and run using
  the set command.

Run command:
  The run command is an arbitrary shell string — no templates or
  placeholders. The LLM provides the exact command based on its full
  context of the project. For Python demos, PYTHONPATH is automatically
  set to the project root if not already present.

Workflow:
  1. LLM creates a demo:
     tinker init parser-demo --lang python
     tinker cell parser-demo "Import and test the new parser"
     tinker code parser-demo <<'EOF'
     from myproject.parser import Parser
     p = Parser()
     result = p.parse("hello")
     assert len(result.tokens) == 1
     print("Parser works:", result)
     EOF
     tinker set-command parser-demo "python .tinker/parser-demo/demo.py"
     tinker run parser-demo

  2. LLM verifies the output (exit code 0 = success)

  3. LLM tells the user:
     "Demo ready at .tinker/parser-demo/demo.py
      Run command: PYTHONPATH=. python .tinker/parser-demo/demo.py"

  4. User opens the file in their editor, navigates cells, and runs
     them interactively to explore the new feature.

Directory structure:
  .tinker/
    <name>/
      demo.<ext>       Demo file (py, c, zig, go, js, sh)
      tinker.toml      Manifest (name, lang, command)

Stdin:
  The cell and code commands accept input from stdin when the text/code
  argument is omitted:
    echo "Test the parser" | tinker cell parser-demo
    cat script.py | tinker code parser-demo

Example — compiled language:
  tinker init sort-bench --lang c
  tinker code sort-bench <<'EOF'
  #include <stdio.h>
  #include <assert.h>

  int main() {
      int arr[] = {3, 1, 2};
      // ... sorting code ...
      assert(arr[0] == 1);
      printf("Sort works!\\n");
      return 0;
  }
  EOF
  tinker set-command sort-bench "cd .tinker/sort-bench && gcc demo.c -o demo && ./demo"
  tinker run sort-bench
"""


def error(msg: str) -> None:
    """Print error to stderr and exit with code 1."""
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from start (or cwd) looking for project markers."""
    current = (start or Path.cwd()).resolve()
    while True:
        for marker in PROJECT_MARKERS:
            if (current / marker).exists():
                return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return (start or Path.cwd()).resolve()


VALID_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


def validate_name(name: str) -> None:
    """Reject demo names that could cause path traversal or other issues."""
    if not VALID_NAME_RE.match(name):
        error(f"invalid demo name '{name}'. Use only letters, numbers, hyphens, and underscores.")


def write_toml(path: Path, data: dict) -> None:
    """Write a TOML file using tomli_w."""
    path.write_bytes(tomli_w.dumps(data).encode())


def read_toml(path: Path) -> dict:
    """Read a TOML file."""
    return tomllib.loads(path.read_text())


def get_demo_dir(root: Path, name: str) -> Path:
    return root / ".tinker" / name


def get_demo_file(root: Path, name: str, lang: str) -> Path:
    ext = LANG_EXT[lang]
    return get_demo_dir(root, name) / f"demo.{ext}"


def get_manifest(root: Path, name: str) -> dict:
    """Read the tinker.toml for a demo. Error if not found."""
    toml_path = get_demo_dir(root, name) / "tinker.toml"
    if not toml_path.exists():
        error(f"demo '{name}' not found. Run 'tinker list' to see available demos.")
    return read_toml(toml_path)


def uses_hash_comments(lang: str) -> bool:
    """Whether the language uses # for comments (python, bash)."""
    return lang in ("python", "bash")


def cmd_init(args, root: Path) -> None:
    name = args.name
    lang = args.lang

    validate_name(name)

    if lang not in SUPPORTED_LANGS:
        error(f"unsupported language '{lang}'. Supported: {', '.join(SUPPORTED_LANGS)}")

    demo_dir = get_demo_dir(root, name)
    if demo_dir.exists():
        error(f"demo '{name}' already exists. Use a different name.")

    demo_dir.mkdir(parents=True)

    ext = LANG_EXT[lang]
    demo_file = demo_dir / f"demo.{ext}"

    # Write header
    if lang == "python":
        demo_file.write_text(f"# %% [markdown]\n# # {name}\n# *Created by tinker*\n\n")
    elif uses_hash_comments(lang):
        demo_file.write_text(f"# {name}\n# Created by tinker\n\n")
    else:
        demo_file.write_text(f"// {name}\n// Created by tinker\n\n")

    # Write manifest
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_toml(demo_dir / "tinker.toml", {
        "name": name,
        "lang": lang,
        "created": now,
        "command": "",
    })

    print(f".tinker/{name}/demo.{ext}")


def cmd_cell(args, root: Path) -> None:
    name = args.name
    manifest = get_manifest(root, name)
    lang = manifest["lang"]
    demo_file = get_demo_file(root, name, lang)

    text = args.text if args.text is not None else sys.stdin.read()

    lines = text.split("\n")

    if lang == "python":
        cell = "# %% [markdown]\n"
        for line in lines:
            cell += f"# {line}\n" if line else "#\n"
    elif uses_hash_comments(lang):
        cell = "# ---\n"
        for line in lines:
            cell += f"# {line}\n" if line else "#\n"
    else:
        cell = "// ---\n"
        for line in lines:
            cell += f"// {line}\n" if line else "//\n"

    cell += "\n"

    with demo_file.open("a") as f:
        f.write(cell)


def cmd_code(args, root: Path) -> None:
    name = args.name
    manifest = get_manifest(root, name)
    lang = manifest["lang"]
    demo_file = get_demo_file(root, name, lang)

    code = args.code if args.code is not None else sys.stdin.read()

    if lang == "python":
        cell = "# %%\n" + code + "\n\n"
    else:
        cell = code + "\n\n"

    with demo_file.open("a") as f:
        f.write(cell)


def cmd_set_command(args, root: Path) -> None:
    name = args.name
    command = args.command
    manifest = get_manifest(root, name)
    lang = manifest["lang"]

    # Auto-prepend PYTHONPATH for Python demos
    if lang == "python" and "PYTHONPATH" not in command:
        command = f"PYTHONPATH={root} {command}"

    # Re-write the manifest with the new command
    manifest["command"] = command
    write_toml(get_demo_dir(root, name) / "tinker.toml", manifest)

    print(command)


def cmd_run(args, root: Path) -> None:
    name = args.name
    manifest = get_manifest(root, name)
    command = manifest.get("command", "")

    if not command:
        error(f"no command set for '{name}'. Run 'tinker set-command {name} <command>'.")

    result = subprocess.run(
        command,
        shell=True,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    sys.exit(result.returncode)


def cmd_show(args, root: Path) -> None:
    name = args.name
    manifest = get_manifest(root, name)
    lang = manifest["lang"]
    demo_file = get_demo_file(root, name, lang)

    print(demo_file.read_text(), end="")

    command = manifest.get("command", "")
    print("\n--- tinker command ---")
    if command:
        print(command)
    else:
        print("[no command set]")


def find_cell_delimiters(lines: list[str], lang: str) -> list[int]:
    """Return line indices of cell delimiters for the given language."""
    if lang == "python":
        return [i for i, line in enumerate(lines) if line.startswith("# %%")]
    delim = "# ---" if uses_hash_comments(lang) else "// ---"
    return [i for i, line in enumerate(lines) if line.strip() == delim]


def find_comment_block_end(lines: list[str], start: int, lang: str) -> int:
    """Find the end of a comment block starting after a delimiter."""
    prefix = "# " if uses_hash_comments(lang) else "// "
    empty_comment = "#" if uses_hash_comments(lang) else "//"
    pos = start
    while pos < len(lines):
        line = lines[pos]
        if line.startswith(prefix) or line == empty_comment:
            pos += 1
        else:
            break
    # Skip trailing blank lines
    while pos < len(lines) and lines[pos].strip() == "":
        pos += 1
    return pos


def cmd_pop(args, root: Path) -> None:
    name = args.name
    manifest = get_manifest(root, name)
    lang = manifest["lang"]
    demo_file = get_demo_file(root, name, lang)

    content = demo_file.read_text()
    lines = content.split("\n")
    delimiter_positions = find_cell_delimiters(lines, lang)

    if lang == "python":
        if len(delimiter_positions) <= 1:
            error(f"cannot pop: only one cell (the header) remains in '{name}'.")
        new_lines = lines[:delimiter_positions[-1]]
    else:
        if not delimiter_positions:
            error(f"cannot pop: only one cell (the header) remains in '{name}'.")

        last_delim = delimiter_positions[-1]
        section_end = find_comment_block_end(lines, last_delim + 1, lang)

        has_trailing_content = any(
            lines[i].strip() for i in range(section_end, len(lines))
        )

        if has_trailing_content:
            new_lines = lines[:section_end]
        else:
            new_lines = lines[:last_delim]

    new_content = "\n".join(new_lines)
    if not new_content.endswith("\n"):
        new_content += "\n"

    demo_file.write_text(new_content)


def cmd_list(args, root: Path) -> None:
    tinker_dir = root / ".tinker"
    if not tinker_dir.exists():
        return

    for entry in sorted(tinker_dir.iterdir()):
        if not entry.is_dir():
            continue
        toml_path = entry / "tinker.toml"
        if not toml_path.exists():
            continue
        manifest = read_toml(toml_path)
        name = manifest.get("name", entry.name)
        lang = manifest.get("lang", "unknown")
        command = manifest.get("command", "")
        if command:
            cmd_display = command[:60] if len(command) <= 60 else command[:57] + "..."
        else:
            cmd_display = "[no command set]"
        print(f"{name} ({lang}) {cmd_display}")


def main():
    # Handle --help and --version before argparse to get exact output format
    if len(sys.argv) == 2 and sys.argv[1] == "--help":
        print(HELP_TEXT, end="")
        sys.exit(0)

    if len(sys.argv) == 2 and sys.argv[1] == "--version":
        print(f"tinker {__version__}")
        sys.exit(0)

    parser = argparse.ArgumentParser(prog="tinker", add_help=False)
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--help", action="store_true")

    subparsers = parser.add_subparsers(dest="subcommand")

    # init
    p_init = subparsers.add_parser("init")
    p_init.add_argument("name")
    p_init.add_argument("--lang", required=True)

    # cell
    p_cell = subparsers.add_parser("cell")
    p_cell.add_argument("name")
    p_cell.add_argument("text", nargs="?", default=None)

    # code
    p_code = subparsers.add_parser("code")
    p_code.add_argument("name")
    p_code.add_argument("code", nargs="?", default=None)

    # set-command
    p_setcmd = subparsers.add_parser("set-command")
    p_setcmd.add_argument("name")
    p_setcmd.add_argument("command")

    # run
    p_run = subparsers.add_parser("run")
    p_run.add_argument("name")

    # show
    p_show = subparsers.add_parser("show")
    p_show.add_argument("name")

    # pop
    p_pop = subparsers.add_parser("pop")
    p_pop.add_argument("name")

    # list
    subparsers.add_parser("list")

    args = parser.parse_args()

    if args.help:
        print(HELP_TEXT, end="")
        sys.exit(0)

    if args.version:
        print(f"tinker {__version__}")
        sys.exit(0)

    if not args.subcommand:
        print(HELP_TEXT, end="")
        sys.exit(0)

    root = find_project_root()

    dispatch = {
        "init": cmd_init,
        "cell": cmd_cell,
        "code": cmd_code,
        "set-command": cmd_set_command,
        "run": cmd_run,
        "show": cmd_show,
        "pop": cmd_pop,
        "list": cmd_list,
    }

    dispatch[args.subcommand](args, root)


if __name__ == "__main__":
    main()
