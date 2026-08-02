<!-- Thanks for contributing to taleTribe-agents! Please fill out the sections below. -->

## Summary

<!-- What does this PR do and why? Link any related issues, e.g. "Closes #123". -->

## Scope of changes

<!-- Bullet the key changes. Call out any new agent actions, API changes, or config/env changes. -->

-

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that changes existing behavior)
- [ ] Documentation / chore (no production code change)

## Verification

<!-- Show how you validated the change. -->

- [ ] `pytest` passes locally
- [ ] Manually tested `GET /health`
- [ ] Manually tested `POST /agent/execute`

```
# paste relevant test output / curl results here
```

## Checklist

- [ ] New agent actions are added to `action_schemas.py` and wired up in `agent.py`
- [ ] New actions have corresponding tests in `tests/`
- [ ] `requirements.txt` and `requirements-prod.txt` are in sync (prod omits ML libs only)
- [ ] No secrets, API keys, or `.env` values are committed
- [ ] Docs updated where relevant (`README.md`, `AGENTS.md`, wiki)
