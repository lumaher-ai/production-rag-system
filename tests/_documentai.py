"""A Document AI stand-in built from the real protos.

Every fake here constructs genuine ``documentai.Document`` messages rather than
duck-typed stubs. That matters more than usual: the thing under test is a
mapping from Google's layout tree onto this system's segments, so a fake with a
convenient shape would verify the mapping against itself. Field names, the
``type_`` rename proto-plus applies, and the fact that table cells contain
*blocks* rather than strings are all part of what can break.

What is faked is only the network call.
"""

from typing import Any

from google.cloud import documentai

Block = documentai.Document.DocumentLayout.DocumentLayoutBlock


def text_block(text: str, block_type: str = "paragraph", page: int = 1, **kw: Any) -> Block:
    return Block(
        text_block=Block.LayoutTextBlock(text=text, type_=block_type, **kw),
        page_span=Block.LayoutPageSpan(page_start=page, page_end=page),
    )


def table_block(
    header: list[str], rows: list[list[str]], caption: str = "", page: int = 1
) -> Block:
    def cell(value: str) -> Block.LayoutTableCell:
        return Block.LayoutTableCell(blocks=[text_block(value, page=page)])

    def row(values: list[str]) -> Block.LayoutTableRow:
        return Block.LayoutTableRow(cells=[cell(v) for v in values])

    return Block(
        table_block=Block.LayoutTableBlock(
            header_rows=[row(header)] if header else [],
            body_rows=[row(r) for r in rows],
            caption=caption,
        ),
        page_span=Block.LayoutPageSpan(page_start=page, page_end=page),
    )


def list_block(items: list[str], page: int = 1) -> Block:
    return Block(
        list_block=Block.LayoutListBlock(
            list_entries=[
                Block.LayoutListEntry(blocks=[text_block(item, page=page)]) for item in items
            ]
        ),
        page_span=Block.LayoutPageSpan(page_start=page, page_end=page),
    )


def document(blocks: list[Block]) -> documentai.Document:
    return documentai.Document(
        document_layout=documentai.Document.DocumentLayout(blocks=blocks)
    )


class FakeDocumentAIClient:
    """Records every request and replies with a canned or generated document.

    ``requests`` is the assertion surface that matters: how many calls were made
    and what page ranges they covered is the difference between correct sharding
    and an expensive accident.
    """

    def __init__(
        self,
        reply: documentai.Document | None = None,
        per_shard: list[documentai.Document] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._reply = reply
        self._per_shard = per_shard
        self._error = error
        self.requests: list[Any] = []

    async def process_document(self, request: Any) -> Any:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        if self._per_shard is not None:
            index = len(self.requests) - 1
            return documentai.ProcessResponse(document=self._per_shard[index])
        return documentai.ProcessResponse(document=self._reply or document([]))

    @staticmethod
    def processor_version_path(project: str, location: str, processor: str, version: str) -> str:
        return (
            f"projects/{project}/locations/{location}/processors/{processor}"
            f"/processorVersions/{version}"
        )

    @property
    def shard_page_counts(self) -> list[int]:
        """Pages in each shard actually sent, read back out of the PDF bytes."""
        from io import BytesIO

        from pypdf import PdfReader

        return [
            len(PdfReader(BytesIO(request.raw_document.content)).pages)
            for request in self.requests
        ]
