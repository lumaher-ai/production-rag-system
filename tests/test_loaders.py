import pytest

from production_rag.exceptions import UnsupportedFileTypeError, ValidationError
from production_rag.ingestion.loaders import load_file
from tests._file_builders import make_docx, make_pdf


def test_pdf_yields_one_segment_per_page() -> None:
    pdf = make_pdf(["First page content here.", "Second page content here."])
    segments = load_file(pdf, "report.pdf", "application/pdf")

    assert [s.page for s in segments] == [1, 2]
    assert "First" in segments[0].text
    assert "Second" in segments[1].text
    assert all(s.section is None for s in segments)


def test_docx_tags_segments_with_headings() -> None:
    docx_bytes = make_docx([("Introduction", "Intro body."), ("Methods", "Methods body.")])
    segments = load_file(
        docx_bytes,
        "paper.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    sections = {s.section for s in segments}
    assert "Introduction" in sections
    assert "Methods" in sections
    assert all(s.page is None for s in segments)


def test_markdown_splits_on_headings() -> None:
    md = b"# Title\nintro line\n\n## Section A\nalpha text\n\n## Section B\nbeta text"
    segments = load_file(md, "notes.md", "text/markdown")

    sections = [s.section for s in segments]
    assert sections == ["Title", "Section A", "Section B"]
    assert "alpha text" in next(s.text for s in segments if s.section == "Section A")


def test_html_converts_and_drops_scripts() -> None:
    html = (
        b"<html><body><h2>Overview</h2><p>Hello world paragraph.</p>"
        b"<script>evil()</script></body></html>"
    )
    segments = load_file(html, "page.html", "text/html")

    assert any(s.section == "Overview" for s in segments)
    assert any("Hello world" in s.text for s in segments)
    assert all("evil()" not in s.text for s in segments)


def test_plain_text_single_segment_no_provenance() -> None:
    segments = load_file(b"just some plain text content", "note.txt", "text/plain")

    assert len(segments) == 1
    assert segments[0].page is None
    assert segments[0].section is None


def test_extension_wins_over_generic_content_type() -> None:
    # Browsers often send application/octet-stream; extension must still resolve.
    md = b"## Heading\nbody text"
    segments = load_file(md, "notes.md", "application/octet-stream")

    assert segments[0].section == "Heading"


def test_unsupported_extension_raises_415() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        load_file(b"data", "archive.xyz", None)


def test_empty_file_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        load_file(b"   \n  ", "empty.txt", "text/plain")


def test_long_section_heading_is_truncated_to_255() -> None:
    heading = "H" * 400
    md = f"## {heading}\nbody".encode()
    segments = load_file(md, "notes.md", "text/markdown")

    assert segments[0].section is not None
    assert len(segments[0].section) == 255
