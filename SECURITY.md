# Security Policy

## Supported versions

Rabbit is pre-1.0. Security fixes land on `main` and in the latest release only.

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Use GitHub's private reporting instead: open the repository's **Security** tab and choose
**Report a vulnerability**. Include the affected version (`./rabbit version`), a description of
the impact, and a minimal reproduction if you have one.

You can expect an acknowledgement within a few days. Fixes are credited in the changelog unless
you ask otherwise.

## Scope notes

Rabbit parses untrusted binary input in two places, and reports there are especially welcome:

- **`.rabbit` model files** (`src/model.c`): header and offset validation, `mmap` bounds.
- **BPE tokenizer files** (`src/tokenizer.c`, format `RBTTOK01`).

The bundled HTTP server and dashboard (`tools/rabbit_server.py`, `tools/rabbit_dashboard.py`) are
meant for local use. They have no authentication, so do not expose them to an untrusted network.
