"""Unit tests for normalize / chunk / content_hash."""

from __future__ import annotations

from openbird.memory import ingest


def test_normalize_collapses_whitespace_and_is_idempotent():
    raw = "Hello   world\t\tfoo  \r\n\r\n\r\n\r\nbar  "
    norm = ingest.normalize(raw)
    assert norm == "Hello world foo\n\nbar"
    assert ingest.normalize(norm) == norm


def test_normalize_empty():
    assert ingest.normalize("") == ""
    assert ingest.normalize("   \n  \t ") == ""


def test_content_hash_stable_under_whitespace_variation():
    a = ingest.content_hash("the  quick   brown\tfox")
    b = ingest.content_hash("the quick brown fox")
    assert a == b
    assert a != ingest.content_hash("the quick brown dog")
    assert len(a) == 64  # sha256 hex


def test_chunk_short_text_single_chunk():
    chunks = ingest.chunk("just a little bit of text")
    assert len(chunks) == 1
    (span, text) = chunks[0]
    assert span == (0, len("just a little bit of text"))
    assert text == "just a little bit of text"


def test_chunk_empty_text():
    assert ingest.chunk("") == []
    assert ingest.chunk("    \n  ") == []


def test_chunk_long_text_multiple_overlapping_chunks():
    sentence = "This is sentence number {}. ".format
    long_text = "".join(sentence(i) for i in range(200))  # well over CHUNK_SIZE
    chunks = ingest.chunk(long_text)
    assert len(chunks) > 1
    # Spans are within bounds and non-empty.
    norm_len = len(ingest.normalize(long_text))
    for (start, end), text in chunks:
        assert 0 <= start < end <= norm_len
        assert text.strip() != ""
    # Coverage: chunks collectively span from near the start to the end.
    assert chunks[0][0][0] == 0
    assert chunks[-1][0][1] == norm_len


def test_chunk_spans_reference_normalized_text():
    text = "alpha beta gamma. " * 100
    norm = ingest.normalize(text)
    for (start, end), ctext in ingest.chunk(text):
        # The stripped slice of normalized text should match the chunk text.
        assert norm[start:end] == ctext
