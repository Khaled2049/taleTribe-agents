# Agents test suite review

Reviewed all 28 test modules and their parameter sets on 2026-10-14, starting from `dc49846` (the chapter-ID fix in PR #39). Collected cases fell from **923 to 601**, a reduction of **322 (35%)**. Counts include parametrized cases, not just test functions. No tests were skipped or excluded from pytest collection, and no production code was changed by this cleanup.

The goal is to protect behavior while allowing implementation and wording to change. Removed duplicate checks, assertions about private structures or exact copy/defaults, primitive framework checks, tests that chiefly exercised their own fakes, and checks that passed for unrelated reasons. A few related assertions were moved into an existing flow; the lower count is not solely regrouping the same checks.

## File-by-file decisions

| Test file | Before | After | Decision |
| --- | ---: | ---: | --- |
| test_action_schemas.py | 8 | 8 | Kept input-size boundaries for the different action payloads. |
| test_agent.py | 9 | 8 | Removed the constructor attribute check; kept dispatch and supported aliases. |
| test_assistant_api.py | 18 | 16 | Removed exact default-value checks and a duplicate no-BYOK test; kept authentication, ownership, streaming, and provider context. |
| test_assistant_contracts.py | 72 | 20 | One wire-contract check per fixture replaces four overlapping passes. Kept manifest, event coverage, and schema drift checks. |
| test_assistant_entity_proposals.py | 64 | 54 | Removed duplicate tool-gating checks, a regex phrase matrix, and an invalid-continuation test that failed on its unrelated userId field. Kept binding, ambiguity, approval, tampering, and unbilled continuations. |
| test_assistant_executors.py | 42 | 21 | Removed private clipping-helper matrices and repeated small-result checks. Kept realistic oversized results, serialized Unicode budgeting, citations, scope, and chapter-ID regressions. |
| test_assistant_help.py | 28 | 2 | Removed copy, fixed catalog composition, and per-capability gating assertions. Kept tool/catalog consistency and JSON delivery. |
| test_assistant_history.py | 10 | 7 | Removed basic replay already covered by the run flow and duplicate no-history/outage cases. Kept interrupted turns, history budgets, and ownership. |
| test_assistant_protocol.py | 25 | 17 | Removed duplicate fixture serialization, primitive string validation, and UID adapter checks. Kept strict identity, compatibility, continuations, and event sequencing; use one future-version case. |
| test_assistant_run.py | 24 | 20 | Removed literal prompt assertions and merged duplicate metadata privacy/history assertions into existing flow tests. Kept billing ceilings, tool rounds, edit approvals, and selection behavior. |
| test_assistant_security.py | 4 | 4 | Kept all four attack scenarios; they exercise the run boundary with adversarial tool calls. |
| test_assistant_specialists.py | 82 | 72 | Removed exact roster/prompt checks and budget helper checks already exercised through consult flows. Fake routing now follows configured prompts instead of requiring literal introductory wording. Kept usage, isolation, concurrent failure, review, drafting, and cancellation. |
| test_assistant_tools.py | 40 | 13 | Removed framework/dataclass introspection, defaults, and unknown-field matrices that often passed because required fields were absent. Check every schema's identity invariant in one registry test; keep valid editor-coordinate rejection and tool bounds. |
| test_embedding_dimension.py | 5 | 3 | Removed no-op success/None cases; kept real vector length and mismatched-dimension behavior. |
| test_embedding_text.py | 9 | 5 | Removed an obsolete hard-coded frontend field list and fallback-copy assertions. Kept chunk coverage/overlap and actual entity text. |
| test_image_generation_routes.py | 4 | 4 | Kept feature gating, authentication, generation concurrency, and safe error handling. |
| test_llm_provider.py | 38 | 25 | Removed retry-decorator introspection and status-helper checks already exercised through HTTP calls. Merged token forwarding with balance retrieval; kept billing no-retry, recovery, request budgets, and streaming. |
| test_mcp_access.py | 20 | 14 | Removed repeated allow/cache cases and exact read-count assertions. Kept denial, outage, cache expiry, and token revocation through tools. |
| test_mcp_blocks.py | 73 | 31 | Replaced two overlapping 24-input matrices with six mixed/malformed document cases checking both preservation and repeated edits. Kept individual parsing and preview regressions. |
| test_mcp_disabled.py | 7 | 5 | Removed a repeated disabled route and a garbage-input test pinning a numeric default; kept disabled service/write behavior and invalid write configuration. |
| test_mcp_oauth_flow.py | 28 | 21 | Merged metadata, valid-token use, and consent-scope assertions into existing flows. Removed a vacuous internal-auth check accepting almost any status and duplicate successful exchanges. Kept PKCE, replay, scopes, denied consent, and family revocation. |
| test_mcp_oauth_store.py | 45 | 26 | Removed hashing/secret-generation helper smoke tests and basic provider lifecycles already exercised over HTTP. Kept expiry, UID/client binding, scope, revocation/mint races, and grant isolation. |
| test_mcp_throttle.py | 23 | 17 | Merged per-IP/write-count checks into registration throttling. Removed framework/version, private IP-helper duplication, and a health check that never exhausted a budget. Kept forged proxy headers, shared/global limits, and streamed-body bounds. |
| test_mcp_tools.py | 174 | 138 | Removed tests primarily validating fake-database quotas/sorting, impossible canonical name types, internal field/constant registries, repeated adapter checks, and a malformed-append test recomputing its expectation with the same helper. Kept read/write flows, privacy, idempotency, contention, stale revisions, original HTML, and service error translation. |
| test_postgres_retrieval.py | 4 | 4 | Kept canonical JSON decoding and bounded chapter-ID roster regressions. |
| test_rate_limit.py | 11 | 7 | Removed private LRU state, an eviction fail-open policy assertion, a wall-clock microbenchmark, and duplicate missing-user validation. Kept observable throttling and bounded tracked keys. |
| test_server.py | 25 | 18 | Removed constructor/framework/duplicate response checks. Retained parameter rejection tests now supply a valid caller, so the target parameter causes rejection. Kept credits and provider forwarding. |
| test_server_auth.py | 31 | 21 | Removed duplicate Settings/create_app checks and repeated generic invalid-environment inputs. Kept production startup failures, caller validation, local-auth refusal in production/Cloud Run, and emulator-token rejection. |

## Retained protection and verification

The remaining tests still exercise ownership (including another user's published story), approval before writes, forged continuations and tool scope, billing retry/idempotency rules, bounded model/tool/context usage, interrupted streams, manuscript preservation, and chapter reads using real identifiers. Security checks at different service boundaries were retained where they enforce different permissions; a lower-level ownership check does not replace the assistant endpoint's ownership check.

All 601 cases pass with `venv/bin/python -m pytest -q`. Black, isort, Ruff, and diff checks also pass.

Six temporary, process-local mutations were checked against retained tests: omitting chapter IDs, bypassing ownership, mutating story data before approval, treating a billing-commit error as retryable, bypassing serialized-result budgeting, and corrupting existing text during append. Each caused its selected regression test to fail. These probes changed no files or external data and are sampled evidence, not a comprehensive mutation-coverage claim.

## Scope and tradeoffs

This intentionally gives up exhaustive primitive input permutations, helper-level snapshots, literal prompt/default checks, and exact internal algorithm choices. Keep a new test when it catches a realistic regression in behavior or a meaningful boundary; do not add a test just to assert a constant or repeat an existing flow at another helper layer.

Most service tests use fakes and mocked providers. A passing suite does not establish live model quality, PostgreSQL isolation, or deployed integration behavior. Story-data's own database/API tests remain the place to verify canonical quotas and ordering. This review did not remove tests from other repositories or change shared assistant fixtures.
