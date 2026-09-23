"""Firestore persistence for the MCP OAuth 2.1 authorization server.

All methods are synchronous (google-cloud-firestore sync client) — the async
provider in oauth_provider.py bridges via anyio.to_thread. Token and code
values are never stored: only their SHA-256 hashes are used as document IDs.

Every load re-checks `expiresAt` in code. Firestore TTL policies (configured
in Terraform on the same field) are garbage collection with a 24-72h lag,
not enforcement.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from google.api_core import exceptions as gcp_exceptions
from google.cloud.firestore_v1.base_query import FieldFilter

CLIENTS_COLLECTION = "mcpOauthClients"
TXNS_COLLECTION = "mcpOauthTxns"
CODES_COLLECTION = "mcpOauthCodes"
TOKENS_COLLECTION = "mcpOauthTokens"
FAMILIES_COLLECTION = "mcpOauthFamilies"
MINT_ATTEMPTS = 3

TXN_STATUS_PENDING = "pending"
TXN_STATUS_COMPLETED = "completed"
TXN_STATUS_DENIED = "denied"

# Internal bookkeeping stripped before a client record is handed to the SDK,
# which validates it against OAuthClientInformationFull.
_CLIENT_INTERNAL_FIELDS = ("createdAt", "expiresAt")


def hash_token(value: str) -> str:
    """SHA-256 hex digest used as the Firestore document ID for codes/tokens."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def new_secret(prefix: str) -> str:
    """Generate an opaque credential value, e.g. mcp_at_<43 urlsafe chars>."""
    return f"{prefix}_{secrets.token_urlsafe(32)}"


def new_family_id() -> str:
    """Id linking every token pair descended from one authorization grant.

    Not a credential — it never leaves the server and is stored in the clear.
    It exists so that detecting a replayed refresh token can revoke the whole
    lineage, including the successor pair the thief is already holding.
    """
    return f"fam_{secrets.token_urlsafe(16)}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: Any) -> Optional[datetime]:
    """Coerce a stored timestamp to an aware UTC datetime; None if absent or odd.

    Firestore hands back tz-aware datetimes, but the emulator and hand-seeded
    test fixtures don't always attach a tzinfo, and comparing naive to aware
    raises. Shared because three call sites now need it (expiry, client TTL
    renewal, and reuse triage) and three copies of the same defensive branch is
    two too many.
    """
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _expired(data: dict[str, Any]) -> bool:
    expires_at = as_utc(data.get("expiresAt"))
    if expires_at is None:
        return False
    return expires_at <= _now()


class OAuthStore:
    """CRUD over the four mcpOauth* collections with expiry checked at read."""

    def __init__(self, db: Any) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Clients (RFC 7591 dynamic registrations)
    # ------------------------------------------------------------------

    def save_client(
        self, client_id: str, record: dict[str, Any], ttl_seconds: int
    ) -> None:
        """Persist a registration with a short initial expiry.

        Registration is unauthenticated (RFC 7591, required by MCP clients), so
        a registration that never completes a flow must not be stored forever.
        The expiry slides out to the full window on first use — see get_client.
        """
        record = dict(record)
        now = _now()
        record["createdAt"] = now
        record["expiresAt"] = now + timedelta(seconds=ttl_seconds)
        self._db.collection(CLIENTS_COLLECTION).document(client_id).set(record)

    def get_client(
        self, client_id: str, *, renew_ttl_seconds: int = 0
    ) -> Optional[dict[str, Any]]:
        ref = self._db.collection(CLIENTS_COLLECTION).document(client_id)
        snap = ref.get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        if _expired(data):
            return None
        if renew_ttl_seconds:
            self._extend_client_past_half_life(ref, data, renew_ttl_seconds)
        for field in _CLIENT_INTERNAL_FIELDS:
            data.pop(field, None)
        return data

    def _extend_client_past_half_life(
        self, ref: Any, data: dict[str, Any], ttl_seconds: int
    ) -> None:
        """Slide a live client's expiry forward, writing only past its half-life.

        Every use of a client goes through get_client, so a naive renewal would
        cost a Firestore write per token refresh. Renewing only in the second
        half of the window keeps an actively used client alive indefinitely for
        at most one write per ttl_seconds/2.
        """
        expires_at = as_utc(data.get("expiresAt"))
        now = _now()
        if expires_at is not None:
            if expires_at - now > timedelta(seconds=ttl_seconds / 2):
                return
        try:
            ref.update({"expiresAt": now + timedelta(seconds=ttl_seconds)})
        except gcp_exceptions.NotFound:
            pass  # concurrently deleted; the caller's copy is still usable

    # ------------------------------------------------------------------
    # Authorization transactions (the /authorize -> consent-page handoff)
    # ------------------------------------------------------------------

    def create_txn(self, data: dict[str, Any], ttl_seconds: int) -> str:
        txn_id = secrets.token_urlsafe(24)
        record = dict(data)
        record["status"] = TXN_STATUS_PENDING
        record["expiresAt"] = _now() + timedelta(seconds=ttl_seconds)
        self._db.collection(TXNS_COLLECTION).document(txn_id).set(record)
        return txn_id

    def get_pending_txn(self, txn_id: str) -> Optional[dict[str, Any]]:
        snap = self._db.collection(TXNS_COLLECTION).document(txn_id).get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        if data.get("status") != TXN_STATUS_PENDING or _expired(data):
            return None
        return data

    def finish_txn(self, txn_id: str, status: str) -> Optional[dict[str, Any]]:
        """Atomically move a pending txn to completed/denied; None if unusable.

        Uses an update-time precondition so a txn can only be finished once
        even under concurrent requests.
        """
        ref = self._db.collection(TXNS_COLLECTION).document(txn_id)
        snap = ref.get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        if data.get("status") != TXN_STATUS_PENDING or _expired(data):
            return None
        try:
            ref.update(
                {"status": status},
                option=self._db.write_option(last_update_time=snap.update_time),
            )
        except (gcp_exceptions.FailedPrecondition, gcp_exceptions.NotFound):
            return None
        return data

    # ------------------------------------------------------------------
    # Authorization codes (single-use)
    # ------------------------------------------------------------------

    def save_code(self, code_hash: str, data: dict[str, Any], ttl_seconds: int) -> None:
        record = dict(data)
        record["expiresAt"] = _now() + timedelta(seconds=ttl_seconds)
        self._db.collection(CODES_COLLECTION).document(code_hash).set(record)

    def load_code(self, code_hash: str) -> Optional[dict[str, Any]]:
        snap = self._db.collection(CODES_COLLECTION).document(code_hash).get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        if _expired(data):
            return None
        return data

    def consume_code(self, code_hash: str) -> Optional[dict[str, Any]]:
        """Load-and-delete with an update-time precondition: exactly one caller wins."""
        ref = self._db.collection(CODES_COLLECTION).document(code_hash)
        snap = ref.get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        try:
            ref.delete(option=self._db.write_option(last_update_time=snap.update_time))
        except (gcp_exceptions.FailedPrecondition, gcp_exceptions.NotFound):
            return None
        if _expired(data):
            return None
        return data

    # ------------------------------------------------------------------
    # Access / refresh tokens (opaque, hashed, paired)
    # ------------------------------------------------------------------

    def save_token_pair(
        self,
        *,
        access_hash: str,
        refresh_hash: str,
        uid: str,
        client_id: str,
        scopes: list[str],
        family_id: str,
        access_ttl_seconds: int,
        refresh_ttl_seconds: int,
    ) -> bool:
        family_ref = self._db.collection(FAMILIES_COLLECTION).document(family_id)
        tokens = self._db.collection(TOKENS_COLLECTION)
        for _ in range(MINT_ATTEMPTS):
            family = family_ref.get()
            if family.exists and (family.to_dict() or {}).get("revoked"):
                return False
            now = _now()
            family_fields = {
                "uid": uid,
                "clientId": client_id,
                "revoked": False,
                "expiresAt": now + timedelta(seconds=refresh_ttl_seconds),
            }
            base = {
                "uid": uid,
                "clientId": client_id,
                "scopes": scopes,
                "familyId": family_id,
                "revoked": False,
            }
            batch = self._db.batch()
            if family.exists:
                batch.update(
                    family_ref,
                    family_fields,
                    option=self._db.write_option(last_update_time=family.update_time),
                )
            else:
                batch.create(family_ref, family_fields)
            batch.set(
                tokens.document(access_hash),
                {
                    **base,
                    "type": "access",
                    "pairedWith": refresh_hash,
                    "expiresAt": now + timedelta(seconds=access_ttl_seconds),
                },
            )
            batch.set(
                tokens.document(refresh_hash),
                {
                    **base,
                    "type": "refresh",
                    "pairedWith": access_hash,
                    "expiresAt": now + timedelta(seconds=refresh_ttl_seconds),
                },
            )
            try:
                batch.commit()
                return True
            except (
                gcp_exceptions.AlreadyExists,
                gcp_exceptions.FailedPrecondition,
                gcp_exceptions.NotFound,
            ):
                continue
        return False

    def family_revoked(self, family_id: str) -> bool:
        if not family_id:
            return False
        snap = self._db.collection(FAMILIES_COLLECTION).document(family_id).get()
        return bool(snap.exists and (snap.to_dict() or {}).get("revoked"))

    def get_token(
        self, token_hash: str, *, include_revoked: bool = False
    ) -> Optional[dict[str, Any]]:
        """Load a token record; None when it is unusable.

        `include_revoked` exists for exactly one caller: refresh-token reuse
        detection needs to tell "no such token" (a guess, or garbage) apart from
        "a token we issued and have already rotated away" (a replay). Every
        other caller wants the default, where a revoked token is simply gone.
        Expiry is never overridden — an expired token carries no signal, because
        a replay after expiry is indistinguishable from a slow client.
        """
        snap = self._db.collection(TOKENS_COLLECTION).document(token_hash).get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        if _expired(data):
            return None
        if include_revoked:
            return data
        if data.get("revoked") or self.family_revoked(data.get("familyId") or ""):
            return None
        return data

    def revoke_token_family(self, family_id: str, retain_seconds: int) -> int:
        """Revoke every still-live token descended from one authorization.

        Returns the number actually revoked. Already-revoked documents are left
        alone: they are the audit trail that reuse detection depends on, and
        rewriting them would cost a write per rotation for no gain. In the
        normal case this kills exactly the current pair, since each rotation
        already revokes its predecessor.
        """
        if not family_id:
            return 0
        now = _now()
        self._db.collection(FAMILIES_COLLECTION).document(family_id).set(
            {
                "revoked": True,
                "revokedAt": now,
                "expiresAt": now + timedelta(seconds=retain_seconds),
            },
            merge=True,
        )
        tokens = self._db.collection(TOKENS_COLLECTION)
        revoked = 0
        for snap in tokens.where(
            filter=FieldFilter("familyId", "==", family_id)
        ).stream():
            if (snap.to_dict() or {}).get("revoked"):
                continue
            try:
                tokens.document(snap.id).update({"revoked": True, "revokedAt": _now()})
                revoked += 1
            except gcp_exceptions.NotFound:
                pass  # collected by TTL between the read and the write
        return revoked

    def consume_refresh_token(self, token_hash: str) -> Optional[dict[str, Any]]:
        """Atomically revoke a live refresh token for rotation; None if unusable.

        The paired access token is revoked best-effort alongside it.
        """
        ref = self._db.collection(TOKENS_COLLECTION).document(token_hash)
        snap = ref.get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        if data.get("type") != "refresh" or data.get("revoked") or _expired(data):
            return None
        try:
            ref.update(
                {"revoked": True, "revokedAt": _now()},
                option=self._db.write_option(last_update_time=snap.update_time),
            )
        except (gcp_exceptions.FailedPrecondition, gcp_exceptions.NotFound):
            return None
        self._revoke_quietly(data.get("pairedWith"))
        return data

    def revoke_token(self, token_hash: str) -> None:
        """RFC 7009 revocation: revoke the token and its pair (best-effort)."""
        snap = self._db.collection(TOKENS_COLLECTION).document(token_hash).get()
        if not snap.exists:
            return
        self._revoke_quietly(token_hash)
        self._revoke_quietly(snap.to_dict().get("pairedWith"))

    def _revoke_quietly(self, token_hash: Optional[str]) -> None:
        """Revoke without caring whether the document is still there.

        `revokedAt` is stamped alongside every revocation because reuse triage
        reads it: a replay seconds after rotation is a client retrying a lost
        response, one hours later is a stolen token. Without the timestamp the
        alert says something happened but not whether to care.
        """
        if not token_hash:
            return
        try:
            self._db.collection(TOKENS_COLLECTION).document(token_hash).update(
                {"revoked": True, "revokedAt": _now()}
            )
        except gcp_exceptions.NotFound:
            pass
