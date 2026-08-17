# Repository Guidelines — taleTribe-agents

This FastAPI service runs NovelSync’s AI story workflows. It calls LLMs through
creditProxy and uses story-data as the canonical source for migrated stories
and pgvector context.

## Commands

- `source venv/bin/activate && python server.py`: run locally on port 8000.
- `source venv/bin/activate && pytest`: run tests.
- `docker build --platform=linux/amd64 -t <image> .`: build the Cloud Run
  image.
- For the integrated stack, use `../story/dev-new.sh` rather than starting
  dependencies independently.

## Integration rules

- Route all LLM calls through `CREDIT_PROXY_URL`; do not call a paid provider
  directly for platform-funded work.
- With `STORY_DATA_DATABASE_URL` set, PostgreSQL is canonical for story context
  and `INDEXING_WORKER_ENABLED=true` consumes the durable indexing outbox.
- Preserve the legacy Firestore path only for non-migrated stories/features.
  Do not introduce Firestore writes for a story-data-owned domain.
- pgvector embeddings are 768 dimensions. Validate dimensions before writing.
- Metadata written to JSONB must be JSON-serializable; convert database values
  such as `Decimal`, UUIDs, and timestamps deliberately.
- New actions require schemas, agent wiring, tests, and safe user-visible error
  handling.

## Configuration and security

- Use environment variables and secret managers for credentials; never commit
  `.env`, service-account keys, or provider API keys.
- Local integrated values include `CREDIT_PROXY_URL=http://localhost:8090`,
  Firestore emulator `localhost:8080`, and story-data PostgreSQL on port 5433.
- Production deployment remains GitHub Actions plus Terraform/Cloud Run.

## Verification

Run focused tests and `pytest`, then check `GET /health`. For changes to the
PostgreSQL pipeline, verify one outbox event is embedded and persisted without
errors.
