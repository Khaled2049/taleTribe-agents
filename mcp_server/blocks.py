from __future__ import annotations

from html.parser import HTMLParser
from typing import NamedTuple

# Elements that never open a nesting level, in either spelling the editor
# emits: TipTap writes bare `<hr>` and `<img ...>` but serialisers upstream of
# it have written `<img />`. Counting either as a level would swallow the rest
# of the chapter into one block.
VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)

# Tag reported for a run of top-level text that no element wraps.
TEXT_TAG = "text"

BLOCK_PREVIEW_CHARS = 120

# Inter-block separator. Matches writes._to_paragraph_html, so content this
# module rebuilds is spelled the same way content it creates is.
BLOCK_SEPARATOR = "\n"


class Block(NamedTuple):
    """One top-level block. `html` is an exact slice of the source string."""

    tag: str
    html: str


def _line_starts(content: str) -> list[int]:
    """Offset of the first character of each line, for getpos() conversion."""
    starts = [0]
    for index, char in enumerate(content):
        if char == "\n":
            starts.append(index + 1)
    return starts


class _Splitter(HTMLParser):
    """Records the offset at which each top-level block begins.

    Only start offsets are collected: HTMLParser reports where a token *starts*
    but not where it ends, and reconstructing the end of a closing tag by
    scanning for '>' is wrong the moment an attribute contains one
    (`<img alt="a>b">`). Each block therefore runs to the start of the next,
    which needs no such guess.
    """

    def __init__(self, content: str) -> None:
        # convert_charrefs=False is load-bearing. With it on, the parser buffers
        # adjacent data and charref tokens and flushes them as a single event,
        # so getpos() stops marking the start of the text actually handed over
        # and every offset derived from it drifts. Off, each raw token gets its
        # own event at its own position.
        super().__init__(convert_charrefs=False)
        self._content = content
        self._line_starts = _line_starts(content)
        self._depth = 0
        self._text_open = False
        self.cuts: list[tuple[int, str]] = []

    def _offset(self) -> int:
        line, col = self.getpos()
        if line - 1 >= len(self._line_starts):
            return len(self._content)
        return min(self._line_starts[line - 1] + col, len(self._content))

    def _begin_element(self, tag: str) -> None:
        if self._depth == 0:
            self.cuts.append((self._offset(), tag))
            self._text_open = False

    def _begin_text(self, offset: int) -> None:
        # Consecutive top-level text tokens coalesce: "a &amp; b" is one block,
        # not three.
        if self._depth == 0 and not self._text_open:
            self.cuts.append((offset, TEXT_TAG))
            self._text_open = True

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self._begin_element(tag)
        if tag not in VOID_TAGS:
            self._depth += 1

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        self._begin_element(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_TAGS:
            return
        if self._depth == 0:
            # A close with nothing open. Keep it as text rather than dropping
            # it — this module must never lose a byte, however malformed.
            self._begin_text(self._offset())
            return
        self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._depth or self._text_open:
            return
        if not data.strip():
            # Whitespace between top-level blocks is a separator, not content.
            return
        # Start the block at the first real character, so the separator that
        # precedes it is not glued onto the front of the slice.
        self._begin_text(self._offset() + (len(data) - len(data.lstrip())))

    def handle_entityref(self, name: str) -> None:
        self._begin_text(self._offset())

    def handle_charref(self, name: str) -> None:
        self._begin_text(self._offset())

    def handle_comment(self, data: str) -> None:
        self._begin_text(self._offset())

    def handle_decl(self, decl: str) -> None:
        self._begin_text(self._offset())

    def handle_pi(self, data: str) -> None:
        self._begin_text(self._offset())


def split_blocks(content: str) -> list[Block]:
    """Split stored chapter HTML into its top-level blocks.

    Returns [] for empty or whitespace-only content — the state
    StoriesRepo.addChapter seeds a new chapter in, and the state a chapter
    reaches if every block is removed.
    """
    if not isinstance(content, str) or not content.strip():
        return []

    parser = _Splitter(content)
    parser.feed(content)
    parser.close()

    blocks: list[Block] = []
    cuts = parser.cuts
    for index, (start, tag) in enumerate(cuts):
        end = cuts[index + 1][0] if index + 1 < len(cuts) else len(content)
        # Only trailing whitespace is stripped, and by construction that run
        # sits at depth 0 between two blocks (or at the very end of the
        # document), never inside one.
        html = content[start:end].rstrip()
        if html:
            blocks.append(Block(tag=tag, html=html))
    return blocks


class _TextExtractor(HTMLParser):
    """Collects the text content of a fragment, tags discarded."""

    def __init__(self) -> None:
        # convert_charrefs=True here: this path wants readable text, so
        # entities should arrive already decoded. Positions are irrelevant, so
        # the buffering that made it unusable in _Splitter costs nothing.
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def block_preview(block_html: str, limit: int = BLOCK_PREVIEW_CHARS) -> str:
    """A short plain-text summary of one block, for addressing it by index.

    Never a regex over tags: `<img alt="a>b">` ends the tag at the wrong '>'
    and leaks markup into what is supposed to be prose. Entities are decoded
    exactly once (the parser does it), because decoding twice would turn a
    literal "&amp;lt;" the author typed into "<".

    Returns "" for blocks with no text of their own (hr, img), which is the
    honest answer — the tag name is what identifies those.
    """
    if not isinstance(block_html, str) or not block_html:
        return ""
    extractor = _TextExtractor()
    extractor.feed(block_html)
    extractor.close()
    text = " ".join("".join(extractor.parts).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def join_blocks(parts: list[str]) -> str:
    """Rebuild chapter content from block strings.

    Every block keeps its own bytes; only the separators between them are
    normalised (TipTap's getHTML() emits none, this emits "\\n"), so the first
    MCP edit of an editor-authored chapter can change its total length by a few
    bytes without changing a single block. Word count is recomputed on the
    rebuilt string, so the frontend's own invariant still holds afterwards.
    """
    return BLOCK_SEPARATOR.join(part for part in parts if part)
