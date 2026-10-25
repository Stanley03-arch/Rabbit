## Summary

What does this change, and why? Link the issue if there is one.

## Validation

- [ ] `make check` passes locally
- [ ] New behavior has a test (C unit test, or a C-vs-NumPy parity test)
- [ ] Docs updated if the CLI, file format or converter flags changed
- [ ] `CHANGELOG.md` updated for user-visible changes

## Compatibility

- [ ] The default build is still pure C11 with no new dependencies
- [ ] No model files, binaries or generated artifacts are included
- [ ] File-format changes are documented in `docs/file-format.md` and keep old files loadable (or say why not)
