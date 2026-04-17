# tinker-cli

> Part of the **tinker** toolkit: **tinker-cli** (this repo) · [tinker-nvim](https://github.com/munziu263/tinker-nvim)

A CLI tool that helps AI agents build interactive demo files for users to explore in their editor.

The `--help` text is comprehensive enough to serve as the only documentation an LLM needs.

## Install

```
uvx tinker-cli
```

or

```
pip install tinker-cli
```

## Quick example

```bash
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
tinker show parser-demo
```

The agent builds the demo, runs it to verify correctness, then tells the user the file path. The user opens it in their editor and explores interactively -- stepping through cells, modifying code, and re-running.

## Commands

| Command                            | Description                          |
|------------------------------------|--------------------------------------|
| `tinker init <name> --lang <lang>` | Create a new demo                    |
| `tinker cell <name> [text]`        | Append commentary (text or stdin)    |
| `tinker code <name> [code]`        | Append code (code or stdin)          |
| `tinker set-command <name> <cmd>`  | Set the run command                  |
| `tinker run <name>`                | Execute the demo                     |
| `tinker show <name>`               | Print demo file and command          |
| `tinker pop <name>`                | Remove the last cell                 |
| `tinker list`                      | List all demos in this project       |

Run `tinker --help` for the full reference.

## Supported languages

python, c, zig, go, javascript, bash

## File formats

- **Python** uses percent-format (`# %%` cells) for REPL-based exploration. Each cell is a block of code preceded by a `# %%` marker and optional commentary. This is the same format used by Jupyter, Spyder, and VS Code's interactive window.

- **Compiled languages** (C, Zig, Go) and scripting languages (JavaScript, Bash) use standalone source files with `// ---` commentary delimiters.

## How it works

1. The LLM calls `tinker init` to create a demo file.
2. It adds cells with `tinker cell` and fills them with `tinker code`.
3. It sets the run command with `tinker set-command` and runs the file with `tinker run` to verify everything works.
4. It tells the user the file path.
5. The user opens the file in their editor and steps through the cells interactively.

The demo files are plain source files. No special runtime or notebook format is required.

## Companion: tinker-nvim

[tinker-nvim](https://github.com/munziu263/tinker-nvim) is the Neovim plugin companion. It provides:

- `<leader>rs` to send cells to a REPL (IPython via toggleterm)
- `<leader>rf` to run the file
- `]h` / `[h` to navigate between cells

The CLI produces files; the plugin runs them interactively. They are fully independent -- neither imports nor depends on the other.

## Inspirations

- **Showboat** by Simon Willison -- LLM-as-user CLI design, where the `--help` text is the skill instruction.
- **Solveit** by fast.ai -- exploratory dialog over code dumps.
- **jupytext.nvim** -- plain `.py` files with `# %%` cells, no notebook JSON.

## License

MIT
