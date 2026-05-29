"""Shared constants for the brain memory layers."""

# Cap on documents fetched from Firestore for in-process cosine-similarity
# ranking. Top-k after ranking is typically 3-5, so 200 recent docs is plenty.
MEMORY_FETCH_LIMIT = 200
