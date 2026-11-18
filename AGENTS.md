# Rules for the coding agent (DocuMind)

I am a student building this to learn and to explain it in interviews. So:
- Do ONE phase at a time. Never build features from later phases.
- Keep changes small. Prefer simple, readable code over clever abstractions.
- After each task, give me: (1) files created/changed, (2) a 5-line plain-English
  explanation of how it works, (3) the exact commands to run and test it.
- Never invent library or model names. Check current official docs. If unsure, ask me.
- Never read, print or commit real secrets. Use backend/.env (gitignored) and keep
  backend/.env.example updated with variable NAMES only.
- Python 3.12, FastAPI, type hints, small functions, docstrings for non-obvious logic.
- Add a pytest test for each new backend module.
- Do not modify files outside the current phase's scope.
