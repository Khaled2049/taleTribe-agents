"""PostgreSQL canonical context, pgvector retrieval, and durable index outbox.

This module is deliberately independent of Firestore.  It is enabled only when
``STORY_DATA_DATABASE_URL`` is configured, allowing legacy Firestore stories to
continue using the old pipeline during the cutover.
"""

import json
import logging
import os
import socket
from decimal import Decimal
from typing import Any

import asyncpg

from .embedding_text import _chunk_text, compose_entity_text

logger = logging.getLogger(__name__)


class PostgresStoryContext:
    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or os.getenv("STORY_DATA_DATABASE_URL", "")
        self.pool: asyncpg.Pool | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.dsn)

    async def start(self) -> None:
        if self.enabled and self.pool is None:
            self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def context(self, story_id: str) -> dict[str, Any]:
        """Return the existing StoryContextBuilder-compatible shape."""
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            story = await conn.fetchrow(
                "SELECT id, title, description, author_name, category, target_audience, language, revision, created_at, updated_at FROM stories WHERE id=$1",
                story_id,
            )
            if story is None:
                raise ValueError(f"Story {story_id} not found")
            characters = await conn.fetch(
                "SELECT id,name,age,art_url,soul,personality,voice,backstory,affiliations,notes,revision,created_at,updated_at FROM characters WHERE story_id=$1 ORDER BY name",
                story_id,
            )
            places = await conn.fetch(
                "SELECT id,name,image_url,description,atmosphere,geography,history,significance,notes,revision,created_at,updated_at FROM places WHERE story_id=$1 ORDER BY name",
                story_id,
            )
            chapters = await conn.fetch(
                'SELECT c.id,c.title,c.content,c.position AS "order",c.word_count,c.revision,c.created_at,c.updated_at,cs.summary FROM chapters c LEFT JOIN chapter_summaries cs ON cs.chapter_id=c.id WHERE c.story_id=$1 ORDER BY c.position',
                story_id,
            )
            lines = await conn.fetch(
                "SELECT id,name,description,revision,created_at,updated_at FROM plot_lines WHERE story_id=$1 ORDER BY created_at",
                story_id,
            )
            events = await conn.fetch(
                "SELECT e.id,e.plot_line_id,e.name,e.content,e.tension_level,e.pacing,e.story_beat,e.emotional_tone,e.notes,e.position,e.chapter_number,e.revision FROM plot_events e JOIN plot_lines l ON l.id=e.plot_line_id WHERE l.story_id=$1 ORDER BY e.position",
                story_id,
            )
        by_line: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            data = _record(event)
            data["orderIndex"] = int(data.pop("position"))
            data["chapterNumber"] = data.pop("chapter_number")
            by_line.setdefault(str(event["plot_line_id"]), []).append(data)
        return {
            "story": _camel(_record(story)),
            "characters": [_camel(_record(x)) for x in characters],
            "places": [_camel(_record(x)) for x in places],
            "chapters": [_camel(_record(x)) for x in chapters],
            "plots": [
                {**_camel(_record(line)), "events": by_line.get(str(line["id"]), [])}
                for line in lines
            ],
        }

    async def retrieve(
        self, story_id: str, embedding: list[float], top_k: int = 4
    ) -> list[dict[str, Any]]:
        assert self.pool is not None
        vector = _vector(embedding)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT source_type,metadata,text FROM story_vector_chunks WHERE story_id=$1 ORDER BY embedding <=> $2::vector LIMIT $3",
                story_id,
                vector,
                top_k,
            )
        return [
            {"kind": row["source_type"], **dict(row["metadata"]), "text": row["text"]}
            for row in rows
        ]

    @staticmethod
    def format_slim_context(context: dict[str, Any]) -> str:
        """Small, bounded roster used by chat alongside vector excerpts."""
        story = context["story"]
        lines = [
            f"Story: {story.get('title', '')}",
            f"Description: {story.get('description', '')}",
        ]
        for label, items in (
            ("Characters", context["characters"]),
            ("Places", context["places"]),
            ("Plot lines", context["plots"]),
            ("Chapters", context["chapters"]),
        ):
            names = [
                str(item.get("name") or item.get("title") or "") for item in items[:12]
            ]
            if names:
                lines.append(f"{label}: " + ", ".join(names))
        return "\n".join(lines)

    async def claim(self, worker: str, limit: int = 20) -> list[asyncpg.Record]:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            return await conn.fetch(
                """WITH next AS (
                     SELECT id FROM indexing_outbox
                     WHERE delivered_at IS NULL AND available_at <= now()
                       AND (lease_expires_at IS NULL OR lease_expires_at < now())
                     ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT $1
                   ) UPDATE indexing_outbox o
                   SET locked_at=now(),locked_by=$2,lease_expires_at=now()+interval '5 minutes',attempts=attempts+1
                   FROM next, stories s WHERE o.id=next.id AND s.id=o.story_id
                   RETURNING o.id,o.aggregate_type,o.aggregate_id,o.story_id,o.operation,o.revision,s.owner_id""",
                limit,
                worker,
            )

    async def consume_index_budget(self, owner_id: str) -> bool:
        """Charge one indexing pass to a story owner's daily ceiling.

        Keyed on the UTC day rather than ``current_date``: the Cloud Functions
        budget this replaces was UTC-keyed, and ``current_date`` would move the
        reset boundary under any deployment whose database is not UTC.

        Applies to BYOK users too — indexing uses the platform embedder whatever
        key a user brings, so the platform pays for it either way.
        """
        assert self.pool is not None
        row = await self.pool.fetchrow(
            """INSERT INTO indexing_usage(user_id,day,pass_count)
               VALUES($1,(now() AT TIME ZONE 'utc')::date,1)
               ON CONFLICT(user_id,day) DO UPDATE SET pass_count=indexing_usage.pass_count+1
               WHERE indexing_usage.pass_count < $2
               RETURNING pass_count""",
            owner_id,
            _index_budget_limit(),
        )
        return row is not None

    async def defer(self, event_id: str, reason: str) -> None:
        """Release an event until the next UTC day without consuming an attempt.

        A budget refusal is not a delivery failure, so it must not burn one of the
        attempts a failure ceiling would count -- otherwise a user at their limit
        would exhaust the retry allowance and lose the event outright.
        """
        assert self.pool is not None
        await self.pool.execute(
            """UPDATE indexing_outbox
               SET locked_at=NULL,locked_by=NULL,lease_expires_at=NULL,attempts=attempts-1,
                   available_at=(date_trunc('day', now() AT TIME ZONE 'utc') + interval '1 day') AT TIME ZONE 'utc',
                   last_error=$2
               WHERE id=$1""",
            event_id,
            reason,
        )

    async def complete(self, event_id: str) -> None:
        assert self.pool is not None
        await self.pool.execute(
            "UPDATE indexing_outbox SET delivered_at=now(),locked_at=NULL,locked_by=NULL,lease_expires_at=NULL,last_error=NULL WHERE id=$1",
            event_id,
        )

    async def fail(self, event_id: str, error: Exception) -> None:
        assert self.pool is not None
        await self.pool.execute(
            "UPDATE indexing_outbox SET locked_at=NULL,locked_by=NULL,lease_expires_at=NULL,available_at=now()+interval '30 seconds',last_error=$2 WHERE id=$1",
            event_id,
            str(error)[:1000],
        )

    async def source(
        self, aggregate_type: str, aggregate_id: str, story_id: str
    ) -> tuple[str, dict[str, Any]] | None:
        """Fetch a canonical source after claim. Missing is a valid delete race."""
        context = await self.context(story_id)
        if aggregate_type == "chapter":
            for item in context["chapters"]:
                if item["id"] == aggregate_id:
                    return item["title"] + "\n\n" + item["content"], {
                        "title": item["title"],
                        "chapterNumber": item.get("order"),
                    }
        if aggregate_type == "character":
            for item in context["characters"]:
                if item["id"] == aggregate_id:
                    return compose_entity_text("character", item), {
                        "name": item["name"]
                    }
        if aggregate_type == "place":
            for item in context["places"]:
                if item["id"] == aggregate_id:
                    return compose_entity_text("place", item), {"name": item["name"]}
        if aggregate_type == "plot_event":
            for line in context["plots"]:
                for item in line["events"]:
                    if item["id"] == aggregate_id:
                        plot_document = {
                            "name": line["name"],
                            "description": line.get("description", ""),
                            "events": [item],
                        }
                        return compose_entity_text("plot", plot_document), {
                            "name": item["name"],
                            "plotLineId": line["id"],
                        }
        return None

    async def replace_chunks(
        self,
        story_id: str,
        kind: str,
        source_id: str,
        revision: int,
        text: str,
        metadata: dict[str, Any],
        embedder,
    ) -> int:
        assert self.pool is not None
        chunks = _chunk_text(text)
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM story_vector_chunks WHERE source_type=$1 AND source_id=$2",
                    kind,
                    source_id,
                )
                for index, chunk in enumerate(chunks):
                    embedding = _vector(await embedder.embed(chunk))
                    await conn.execute(
                        "INSERT INTO story_vector_chunks(id,story_id,source_type,source_id,source_revision,chunk_index,text,metadata,embedding) VALUES(gen_random_uuid(),$1,$2,$3,$4,$5,$6,$7::jsonb,$8::vector)",
                        story_id,
                        kind,
                        source_id,
                        revision,
                        index,
                        chunk,
                        json.dumps(metadata, default=_json_default),
                        embedding,
                    )
        return len(chunks)

    async def delete_chunks(self, kind: str, source_id: str) -> int:
        assert self.pool is not None
        result = await self.pool.execute(
            "DELETE FROM story_vector_chunks WHERE source_type=$1 AND source_id=$2",
            kind,
            source_id,
        )
        return int(result.rsplit(" ", 1)[-1])


class PostgresIndexWorker:
    def __init__(self, store: PostgresStoryContext, embedder):
        self.store, self.embedder = store, embedder
        self.worker_id = f"agents-{socket.gethostname()}-{os.getpid()}"

    async def run_once(self) -> int:
        if not self.store.enabled or self.embedder is None:
            return 0
        claimed = await self.store.claim(self.worker_id)
        superseded = _superseded(claimed)
        for event in claimed:
            event_id = str(event["id"])
            try:
                kind, source_id = event["aggregate_type"], str(event["aggregate_id"])
                if event_id in superseded:
                    pass
                elif event["operation"] == "delete":
                    # Never budgeted: refusing a delete would leave chunks behind
                    # for content the author removed.
                    await self.store.delete_chunks(kind, source_id)
                else:
                    source = await self.store.source(
                        kind, source_id, str(event["story_id"])
                    )
                    if source is None:
                        await self.store.delete_chunks(kind, source_id)
                    else:
                        # Charged after the source resolves, so a delete race
                        # costs nothing, and before the embedder runs, so the
                        # ceiling is enforced ahead of the spend.
                        if not await self.store.consume_index_budget(event["owner_id"]):
                            logger.warning(
                                "indexing_budget_exhausted",
                                extra={"event_id": event_id},
                            )
                            await self.store.defer(
                                event_id, "indexing budget exhausted"
                            )
                            continue
                        text, metadata = source
                        await self.store.replace_chunks(
                            str(event["story_id"]),
                            kind,
                            source_id,
                            event["revision"],
                            text,
                            metadata,
                            self.embedder,
                        )
                await self.store.complete(event_id)
            except Exception as exc:
                logger.exception("indexing_outbox_failed", extra={"event_id": event_id})
                await self.store.fail(event_id, exc)
        return len(claimed)


def _superseded(events: list[asyncpg.Record]) -> set[str]:
    """Event ids outranked by a higher revision of the same source in this batch.

    Nothing collapses the outbox on the write side -- story-data inserts a row per
    save -- so a burst of autosaves arrives as N events that would each re-embed
    the same chapter for only the last result to survive. Dropping the losers is
    what keeps one editing session costing roughly one pass, which is the unit the
    daily budget is denominated in.
    """
    newest: dict[tuple[str, str], asyncpg.Record] = {}
    for event in events:
        key = (event["aggregate_type"], str(event["aggregate_id"]))
        winner = newest.get(key)
        if winner is None or event["revision"] > winner["revision"]:
            newest[key] = event
    kept = {str(event["id"]) for event in newest.values()}
    return {str(event["id"]) for event in events} - kept


def _index_budget_limit() -> int:
    """Daily embedding passes per user. KEEP IN SYNC with MAX_INDEX_USAGE in the
    frontend's Cloud Functions, which still meters the legacy Firestore path."""
    try:
        parsed = int(os.getenv("MAX_INDEX_USAGE", "300"))
    except ValueError:
        return 300
    return parsed if parsed > 0 else 300


def _record(row: asyncpg.Record) -> dict[str, Any]:
    return {
        key: (str(value) if key.endswith("id") and value is not None else value)
        for key, value in dict(row).items()
    }


def _camel(data: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "author_name": "author",
        "art_url": "artUrl",
        "image_url": "imageUrl",
        "target_audience": "targetAudience",
        "word_count": "wordCount",
        "created_at": "createdAt",
        "updated_at": "updatedAt",
    }
    return {aliases.get(key, key): value for key, value in data.items()}


def _vector(values: list[float]) -> str:
    if len(values) != 768:
        raise ValueError(f"pgvector requires 768 dimensions, got {len(values)}")
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def _json_default(value: Any) -> Any:
    """Encode PostgreSQL numeric values without losing integral metadata."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
