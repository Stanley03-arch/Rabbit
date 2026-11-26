## Summary
<!-- 1–3 sentences: what this PR does and why -->

## Type of change
- [ ] Bug fix
- [ ] New feature / enhancement
- [ ] Documentation
- [ ] Tests / CI
- [ ] Refactor (no behavior change)
- [ ] Chore (build, deps, tooling)

## Checklist
- [ ] I ran `make` and the binary builds
- [ ] I ran relevant tests (`make test-model`, `make check`, or the specific target)
- [ ] CHANGELOG.md updated if user-visible behavior changed
- [ ] Docs updated if CLI, API, or converter flags changed
- [ ] This PR is focused (one logical change)

## How to test
```bash
# paste the exact commands you used
make
make test-model
./rabbit chat --model models/tiny-test.rabbit --tokens 8
```

## Notes for reviewers
<!-- Anything tricky, design trade-offs, or follow-ups -->
