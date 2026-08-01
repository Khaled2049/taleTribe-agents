"""Unit tests for mcp_server.blocks — the top-level HTML block splitter.

Pure string tests: no Firestore fake, no MCP plumbing. The fixtures are shaped
like what the frontend's TipTap configuration actually emits (headings capped
at h1-h3, list/task-list markup with data-type attributes, blockquotes wrapping
nested paragraphs, lowlight code blocks, images with custom display attrs).
"""

import pytest

from mcp_server.blocks import (
    BLOCK_PREVIEW_CHARS,
    TEXT_TAG,
    Block,
    block_preview,
    join_blocks,
    split_blocks,
)


def _tags(content: str) -> list[str]:
    return [block.tag for block in split_blocks(content)]


def _htmls(content: str) -> list[str]:
    return [block.html for block in split_blocks(content)]


# ---------------------------------------------------------------------------
# The invariant: blocks are contiguous, in-order slices of the input
# ---------------------------------------------------------------------------

FIXTURES = [
    "",
    "   \n  ",
    "<p>one</p>",
    "<p>one</p><p>two</p>",
    "<p>one</p>\n<p>two</p>",
    "<h1>Title</h1>\n<h2>Sub</h2>\n<h3>Deep</h3>\n<p>body</p>",
    '<ul class="list-disc"><li><p>a</p></li><li><p>b</p></li></ul>',
    '<ol class="list-decimal"><li>one</li><li>two</li></ol>',
    '<ul data-type="taskList"><li data-type="taskItem" data-checked="false">'
    '<label><input type="checkbox"><span></span></label><div><p>todo</p></div>'
    "</li></ul>",
    "<blockquote><p>quoted</p><p>still quoted</p></blockquote><p>after</p>",
    '<pre><code class="language-python">if a &lt; b:\n    pass</code></pre>',
    "<p>before</p><hr><p>after</p>",
    '<p>a</p><img src="x.png" data-display-mode="wrap" data-align="left"><p>b</p>',
    '<p>a</p><img src="x.png" alt="a>b" /><p>c</p>',
    "<p>line one<br>line two</p>",
    "bare text",
    "bare text<p>then a block</p>",
    "<p>block</p>\n  trailing bare text",
    "<p>unclosed",
    "</p><p>real</p>",
    "<p>a &amp; b</p>",
    "a &amp; b",
    "<p>x</p><!-- a comment --><p>y</p>",
    '<p style="text-align: center"><span style="color: #f00">red</span></p>',
]


@pytest.mark.parametrize("content", FIXTURES)
def test_blocks_are_contiguous_in_order_slices(content):
    """The load-bearing property: nothing is reserialised, nothing is lost.

    Each block must be findable in the source at a position after the previous
    one, and the only characters between consecutive blocks may be whitespace —
    the depth-0 separators the splitter deliberately strips.
    """
    blocks = split_blocks(content)
    cursor = 0
    for block in blocks:
        start = content.find(block.html, cursor)
        assert start >= 0, f"block {block.html!r} is not a slice of the source"
        assert not content[
            cursor:start
        ].strip(), f"non-whitespace dropped between blocks: {content[cursor:start]!r}"
        cursor = start + len(block.html)
    assert not content[
        cursor:
    ].strip(), f"trailing content dropped: {content[cursor:]!r}"


@pytest.mark.parametrize("content", FIXTURES)
def test_split_is_stable_under_rejoin(content):
    """Re-splitting rebuilt content yields the same blocks.

    join_blocks normalises separators, so a second edit of an already-edited
    chapter must not re-partition it differently.
    """
    once = split_blocks(content)
    twice = split_blocks(join_blocks([block.html for block in once]))
    assert once == twice


# ---------------------------------------------------------------------------
# Element blocks
# ---------------------------------------------------------------------------


def test_empty_and_whitespace_only_content():
    assert split_blocks("") == []
    assert split_blocks("   \n\t ") == []
    assert split_blocks(None) == []  # type: ignore[arg-type]


def test_paragraph_sequence_with_and_without_separators():
    # TipTap emits no separator; content this service writes uses "\n".
    assert _htmls("<p>one</p><p>two</p>") == ["<p>one</p>", "<p>two</p>"]
    assert _htmls("<p>one</p>\n<p>two</p>") == ["<p>one</p>", "<p>two</p>"]


def test_headings_are_separate_blocks():
    content = "<h1>A</h1><h2>B</h2><h3>C</h3>"
    assert _tags(content) == ["h1", "h2", "h3"]


def test_list_is_one_block_not_one_per_item():
    content = '<ul class="list-disc"><li><p>a</p></li><li><p>b</p></li></ul>'
    blocks = split_blocks(content)
    assert len(blocks) == 1
    assert blocks[0] == Block(tag="ul", html=content)


def test_task_list_attributes_do_not_split():
    content = (
        '<ul data-type="taskList">'
        '<li data-type="taskItem" data-checked="true">'
        '<label><input type="checkbox"><span></span></label><div><p>done</p></div>'
        "</li></ul>"
    )
    assert split_blocks(content) == [Block(tag="ul", html=content)]


def test_blockquote_with_nested_paragraphs_is_one_block():
    """Depth tracking: the nested <p>s are inside, not siblings."""
    quote = "<blockquote><p>quoted</p><p>still</p></blockquote>"
    blocks = split_blocks(quote + "<p>after</p>")
    assert [b.tag for b in blocks] == ["blockquote", "p"]
    assert blocks[0].html == quote


def test_code_block_entities_do_not_open_blocks():
    """Regression guard for convert_charrefs=False.

    Entities arrive as their own parser events; if one were treated as the
    start of a top-level block, a code sample would shatter mid-line.
    """
    content = '<pre><code class="language-python">if a &lt; b:\n    pass</code></pre>'
    assert split_blocks(content) == [Block(tag="pre", html=content)]


# ---------------------------------------------------------------------------
# Void elements
# ---------------------------------------------------------------------------


def test_bare_void_elements_are_their_own_blocks():
    assert _tags("<p>a</p><hr><p>b</p>") == ["p", "hr", "p"]
    assert _tags('<p>a</p><img src="x.png"><p>b</p>') == ["p", "img", "p"]


def test_self_closed_void_spelling():
    assert _tags('<p>a</p><img src="x.png" /><p>b</p>') == ["p", "img", "p"]
    assert _tags("<hr />") == ["hr"]


def test_br_inside_a_paragraph_does_not_split():
    content = "<p>line one<br>line two</p>"
    assert split_blocks(content) == [Block(tag="p", html=content)]


def test_attribute_containing_a_gt_does_not_truncate_the_block():
    """The reason block ends are cut points, not a scan for '>'."""
    content = '<img src="x.png" alt="a>b"><p>after</p>'
    blocks = split_blocks(content)
    assert [b.tag for b in blocks] == ["img", "p"]
    assert blocks[0].html == '<img src="x.png" alt="a>b">'


# ---------------------------------------------------------------------------
# Bare top-level text
# ---------------------------------------------------------------------------


def test_bare_text_alone():
    assert split_blocks("bare text") == [Block(tag=TEXT_TAG, html="bare text")]


def test_bare_text_between_elements():
    blocks = split_blocks("<p>a</p>middle<p>b</p>")
    assert [b.tag for b in blocks] == ["p", TEXT_TAG, "p"]
    assert blocks[1].html == "middle"


def test_leading_separator_is_not_glued_to_a_text_block():
    blocks = split_blocks("<p>a</p>\n   trailing words")
    assert blocks[1].html == "trailing words"


def test_consecutive_text_tokens_coalesce():
    """'a &amp; b' is one text block, not data/entity/data as three."""
    blocks = split_blocks("a &amp; b")
    assert blocks == [Block(tag=TEXT_TAG, html="a &amp; b")]


# ---------------------------------------------------------------------------
# Malformed input: lenient, never lossy
# ---------------------------------------------------------------------------


def test_unclosed_element_runs_to_end():
    blocks = split_blocks("<p>a</p><p>unclosed and then some")
    assert [b.tag for b in blocks] == ["p", "p"]
    assert blocks[1].html == "<p>unclosed and then some"


def test_stray_end_tag_is_preserved_as_text():
    blocks = split_blocks("</p><p>real</p>")
    assert [b.tag for b in blocks] == [TEXT_TAG, "p"]
    assert blocks[0].html == "</p>"


def test_stray_end_tag_does_not_drive_depth_negative():
    """A negative depth would make every later block look nested and vanish."""
    blocks = split_blocks("</div></div><p>a</p><p>b</p>")
    assert [b.tag for b in blocks][-2:] == ["p", "p"]


def test_comment_at_top_level_is_kept():
    blocks = split_blocks("<p>x</p><!-- note --><p>y</p>")
    assert "".join(b.html for b in blocks).count("<!-- note -->") == 1


# ---------------------------------------------------------------------------
# block_preview
# ---------------------------------------------------------------------------


def test_preview_strips_tags_and_decodes_entities_once():
    assert block_preview("<p>hello <strong>world</strong></p>") == "hello world"
    # A literal "&lt;" the author typed must not decode twice into "<".
    assert block_preview("<p>&amp;lt;</p>") == "&lt;"


def test_preview_collapses_whitespace():
    assert block_preview("<p>a\n\n   b\tc</p>") == "a b c"


def test_preview_truncates_with_an_ellipsis():
    long = "word " * 200
    preview = block_preview(f"<p>{long}</p>")
    assert len(preview) <= BLOCK_PREVIEW_CHARS
    assert preview.endswith("…")


def test_preview_of_textless_blocks_is_empty():
    assert block_preview("<hr>") == ""
    assert block_preview('<img src="x.png" alt="ignored">') == ""
    assert block_preview("") == ""


def test_preview_of_a_gt_in_an_attribute_does_not_leak_markup():
    assert block_preview('<p><img src="x" alt="a>b">text</p>') == "text"


# ---------------------------------------------------------------------------
# join_blocks
# ---------------------------------------------------------------------------


def test_join_uses_the_paragraph_separator_and_drops_empties():
    assert join_blocks(["<p>a</p>", "<p>b</p>"]) == "<p>a</p>\n<p>b</p>"
    assert join_blocks(["<p>a</p>", "", "<p>b</p>"]) == "<p>a</p>\n<p>b</p>"
    assert join_blocks([]) == ""
