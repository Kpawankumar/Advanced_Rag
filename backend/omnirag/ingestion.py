"""Native PDF text with bounded OCR fallback in a disposable child process.

Text formats (DOCX/TXT/MD/CSV/JSON) and web pages are extracted synchronously into
the same chunk/page shape so retrieval, citations and deletion stay uniform.
"""

import ipaddress
import json
import re
import socket
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pypdfium2 as pdfium
from pypdf import PdfReader

from omnirag.ocr import OCRError, largest_image_fraction, recognize_page

EXTRACTOR_VERSION = "pdf_text_ocr_v2"
TEXT_EXTRACTOR_VERSION = "text_ingest_v1"
SPARSE_NATIVE_LIMIT = 200
LARGE_IMAGE_FRACTION = 0.5
TEXT_PAGE_CHARACTERS = 2700
URL_FETCH_MAX_REDIRECTS = 4


class IngestionError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_text(text: str, size: int = 900, overlap: int = 120) -> list[str]:
    text = normalize(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def parse_pdf(
    path: Path,
    max_pages: int,
    ocr_enabled: bool = True,
    ocr_max_pages: int = 20,
    ocr_page_timeout_seconds: int = 20,
) -> dict:
    rendered = None
    try:
        reader = PdfReader(path, strict=False)
        if reader.is_encrypted:
            raise IngestionError("ENCRYPTED_PDF", "Encrypted PDFs are not supported.")
        if len(reader.pages) > max_pages:
            raise IngestionError("PAGE_LIMIT", f"This PDF exceeds the {max_pages}-page limit.")
        chunks, empty_pages, total_chars = [], [], 0
        pages, ocr_pages = [], []
        for page_number, page in enumerate(reader.pages, start=1):
            text = normalize(page.extract_text() or "")
            metadata = {
                "page": page_number,
                "method": "native",
                "extractor_version": EXTRACTOR_VERSION,
                "native_characters": len(text),
            }
            trigger = "low_native_text" if len(text) < 20 else None
            if 20 <= len(text) < SPARSE_NATIVE_LIMIT:
                if rendered is None:
                    rendered = pdfium.PdfDocument(path)
                fraction = largest_image_fraction(rendered, page_number - 1)
                metadata["largest_image_fraction"] = round(fraction, 4)
                if fraction >= LARGE_IMAGE_FRACTION:
                    trigger = "sparse_text_large_image"
                    if not ocr_enabled:
                        raise IngestionError(
                            "OCR_REQUIRED",
                            "A page has sparse text over a large image. Enable local OCR "
                            "to extract its possible scanned content.",
                        )
            if trigger and ocr_enabled:
                if len(ocr_pages) >= ocr_max_pages:
                    raise IngestionError(
                        "OCR_PAGE_LIMIT", f"PDF exceeds the {ocr_max_pages}-page OCR limit."
                    )
                if rendered is None:
                    rendered = pdfium.PdfDocument(path)
                recognized = recognize_page(rendered, page_number - 1, ocr_page_timeout_seconds)
                text = normalize(recognized.pop("text"))
                if trigger == "sparse_text_large_image" and len(text) < 20:
                    raise IngestionError(
                        "NO_TEXT", "No usable OCR text found on a mixed-content page."
                    )
                metadata.update(method="ocr", ocr_trigger=trigger, **recognized)
                ocr_pages.append(page_number)
            metadata["characters"] = len(text)
            pages.append(metadata)
            total_chars += len(text)
            if total_chars > 1_000_000:
                raise IngestionError("TEXT_LIMIT", "Extracted text exceeds the pilot corpus limit.")
            if len(text) < 20:
                empty_pages.append(page_number)
            for ordinal, chunk in enumerate(split_text(text)):
                if len(chunk) >= 20:
                    chunks.append({"page": page_number, "ordinal": ordinal, "text": chunk})
        if not chunks:
            raise IngestionError(
                "NO_TEXT" if ocr_enabled else "OCR_REQUIRED",
                "No usable text found after OCR. Check the original PDF."
                if ocr_enabled
                else "No usable text found. Enable local OCR for scanned PDFs.",
            )
        warnings = []
        if empty_pages:
            warnings.append(
                f"Little or no usable text on pages {empty_pages}; these pages are not indexed."
            )
        if ocr_pages:
            warnings.append(f"OCR transcription on pages {ocr_pages}. Verify against the original.")
        return {
            "page_count": len(reader.pages),
            "chunks": chunks,
            "warnings": warnings,
            "pages": pages,
        }
    except OCRError as exc:
        raise IngestionError(exc.code, exc.message) from exc
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError("INVALID_PDF", "The PDF could not be parsed.") from exc
    finally:
        if rendered is not None:
            rendered.close()


def main() -> None:
    try:
        options = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        result = parse_pdf(Path(sys.argv[1]), int(sys.argv[2]), **options)
    except IngestionError as exc:
        result = {"error_code": exc.code, "error_message": exc.message}
    print(json.dumps(result))


# --- Text formats and web pages (merged from the RAG project's ingestion set) ---


def _text_document_result(text: str, warnings: list[str] | None = None) -> dict:
    text = normalize(text)
    if len(text) < 20:
        raise IngestionError("NO_TEXT", "No usable text was found in this document.")
    if len(text) > 1_000_000:
        raise IngestionError("TEXT_LIMIT", "Extracted text exceeds the corpus limit.")
    # Synthetic pages keep page-linked citations meaningful for text documents.
    pages, chunks = [], []
    for page_number, start in enumerate(range(0, len(text), TEXT_PAGE_CHARACTERS), start=1):
        page_text = text[start : start + TEXT_PAGE_CHARACTERS]
        pages.append(
            {
                "page": page_number,
                "method": "native",
                "extractor_version": TEXT_EXTRACTOR_VERSION,
                "characters": len(page_text),
            }
        )
        for ordinal, chunk in enumerate(split_text(page_text)):
            if len(chunk) >= 20:
                chunks.append({"page": page_number, "ordinal": ordinal, "text": chunk})
    return {"page_count": len(pages), "chunks": chunks, "warnings": warnings or [], "pages": pages}


def parse_docx(path: Path) -> dict:
    try:
        from docx import Document

        document = Document(str(path))
    except Exception as exc:
        raise IngestionError("INVALID_DOCX", "The DOCX file could not be parsed.") from exc
    parts = [para.text for para in document.paragraphs if para.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return _text_document_result("\n".join(parts))


def parse_text_file(path: Path, suffix: str) -> dict:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise IngestionError("READ_FAILED", "The file could not be read as text.") from exc
    if suffix == ".json":
        try:
            text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except ValueError:
            raise IngestionError("INVALID_JSON", "The JSON file could not be parsed.") from None
    return _text_document_result(text)


def parse_text_document(path: Path, suffix: str) -> dict:
    dispatch = {
        ".docx": lambda: parse_docx(path),
        ".txt": lambda: parse_text_file(path, suffix),
        ".md": lambda: parse_text_file(path, suffix),
        ".csv": lambda: parse_text_file(path, suffix),
        ".json": lambda: parse_text_file(path, suffix),
    }
    handler = dispatch.get(suffix.lower())
    if handler is None:
        raise IngestionError("UNSUPPORTED_TYPE", f"Unsupported text format: {suffix}")
    return handler()


def parse_web_page(text: str) -> dict:
    """Chunk already-extracted web page text into the shared document shape."""
    return _text_document_result(text)


class _ReadableText(HTMLParser):
    """Minimal readable-text extractor: drops script/style and keeps block breaks."""

    SKIPPED = {"script", "style", "noscript", "template", "svg", "head"}
    BLOCKS = {
        "p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4",
        "h5", "h6", "br", "pre", "blockquote", "table", "ul", "ol", "header",
        "footer", "nav", "main", "aside", "td", "th",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIPPED:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCKS and not self._skip_depth:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIPPED and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCKS and not self._skip_depth:
            self.parts.append("\n")

    def handle_data(self, data):
        # The title sits inside <head>, which is otherwise skipped content.
        if self._in_title:
            self.title += data
            return
        if self._skip_depth:
            return
        self.parts.append(data)

    def text(self) -> str:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in "".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> tuple[str, str]:
    parser = _ReadableText()
    try:
        parser.feed(html)
    except Exception:
        return "", ""
    return parser.text(), parser.title.strip()


def validate_public_url(url: str) -> str:
    """Allow only plain http(s) URLs to public hosts; blocks SSRF targets."""
    try:
        parts = urlsplit(url.strip())
    except ValueError as exc:
        raise IngestionError("URL_INVALID", "That URL could not be parsed.") from exc
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise IngestionError("URL_INVALID", "Only http(s) URLs with a host are supported.")
    if parts.username or parts.password or (parts.port and not 0 < parts.port <= 65535):
        raise IngestionError("URL_INVALID", "URLs with credentials or unusual ports are rejected.")
    try:
        candidates = socket.getaddrinfo(parts.hostname, None)
    except OSError as exc:
        raise IngestionError("URL_UNREACHABLE", "The host could not be resolved.") from exc
    for info in candidates:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise IngestionError(
                "URL_BLOCKED", "Local, private or reserved network addresses are not fetched."
            )
    return url.strip()


def fetch_url_text(
    url: str, timeout_seconds: int = 20, max_mb: int = 5
) -> tuple[str, str, str]:
    """Fetch a public web page and return (text, title, final_url).

    Redirects are followed manually so every hop passes the same SSRF check.
    """
    current = validate_public_url(url)
    limit = max_mb * 1024 * 1024
    with httpx.Client(
        headers={"User-Agent": "OmniRAG/0.5 (local document workspace)"},
        follow_redirects=False,
        trust_env=False,
        timeout=timeout_seconds,
    ) as client:
        for _ in range(URL_FETCH_MAX_REDIRECTS + 1):
            current = validate_public_url(current)
            response = client.get(current)
            if response.is_redirect:
                location = response.headers.get("location", "")
                if not location:
                    break
                current = str(response.next_request.url) if response.next_request else location
                continue
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if response.status_code == 200 and not any(
                kind in content_type.lower() for kind in ("html", "text/", "xml", "json")
            ):
                raise IngestionError(
                    "URL_TYPE", "Only text, HTML, XML or JSON pages can be ingested."
                )
            if len(response.content) > limit:
                raise IngestionError("URL_TOO_LARGE", "The page exceeds the fetch size limit.")
            kinds = content_type.lower()
            if any(kind in kinds for kind in ("json", "xml")) and "html" not in kinds:
                text = response.text
                title = ""
            else:
                text, title = html_to_text(response.text)
            return text, title, str(response.url)
    raise IngestionError("URL_REDIRECTS", "Too many redirects while fetching the page.")


if __name__ == "__main__":
    main()
