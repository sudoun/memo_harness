# Contributing

Thanks for helping improve Posterior Memory Harness.

## Local setup

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -q
```

The supported Python baseline is 3.10. Keep changes compatible with all
versions covered by CI.

## Pull requests

- Keep each pull request focused and explain the user-visible behavior.
- Add or update tests for behavior changes.
- Run the test command above and ensure `python -m build` succeeds.
- Do not commit local databases, model weights, build artifacts, or generated
  verification environments.
- Report security-sensitive issues through the process in [SECURITY.md](SECURITY.md),
  not in a public issue.
