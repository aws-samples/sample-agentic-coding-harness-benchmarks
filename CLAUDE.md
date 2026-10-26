# Claude Coding Rules

The coding standards and agent instructions for this repository live in **[AGENTS.md](AGENTS.md)**, the single canonical guide every coding agent follows (Claude Code, opencode, Codex, Cursor, Aider and others).

Claude Code reads this file rather than AGENTS.md, so the line below imports it. The import pulls in the repository navigation rules, the paths not to read, the Markdown conventions, and the Git rules -- including **never commit directly to `main`** and **never merge a PR without review**, which Claude Code could not see while this file carried only a Markdown note. Keep substantive rules in AGENTS.md so the two never drift.

@AGENTS.md
