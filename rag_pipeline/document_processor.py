"""
document_processor.py
======================
Handles loading and pre-processing of PDF, text, and Markdown documents.

Supported backends
------------------
- pymupdf   (fitz)       : fast text + image extraction
- pdfplumber             : precise table + text extraction
- nougat                 : scientific PDF → LaTeX / Markdown (formula-aware)
- unstructured           : mixed-layout documents (HTML, tables, images)

All heavy imports are wrapped in try/except so the module loads even when
optional libraries are absent; a helpful install message is printed instead.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional library imports
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("[DocumentProcessor] pymupdf not installed. Run: pip install pymupdf")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("[DocumentProcessor] pdfplumber not installed. Run: pip install pdfplumber")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[DocumentProcessor] Pillow not installed. Run: pip install Pillow")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("[DocumentProcessor] pytesseract not installed. Run: pip install pytesseract")

try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("[DocumentProcessor] unstructured not installed. Run: pip install unstructured")

try:
    import nougat  # nougat-ocr
    NOUGAT_AVAILABLE = True
except ImportError:
    NOUGAT_AVAILABLE = False
    # nougat is optional — only needed for scientific PDFs
    pass


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PageContent:
    """Represents the extracted content of a single PDF page."""
    page_num: int
    text: str
    tables: List[Dict[str, Any]] = field(default_factory=list)   # [{headers, rows, raw_text}]
    images: List[str] = field(default_factory=list)              # saved image file paths
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """Represents a fully parsed document ready for chunking."""
    source_path: str
    pages: List[PageContent] = field(default_factory=list)
    total_pages: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenate all page texts into one string."""
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


# ---------------------------------------------------------------------------
# PDFParser
# ---------------------------------------------------------------------------

class PDFParser:
    """
    Multi-backend PDF parser.  Selects the most appropriate backend
    automatically, or uses the one specified in RAGConfig.pdf_model.
    """

    def __init__(self, config: "RAGConfig"):
        self.config = config
        os.makedirs(config.image_output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Backend: PyMuPDF
    # ------------------------------------------------------------------

    def parse_with_pymupdf(self, pdf_path: str) -> ProcessedDocument:
        """Extract text, images and rough table regions using PyMuPDF (fitz)."""
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("pymupdf is not installed. Run: pip install pymupdf")

        doc = fitz.open(pdf_path)
        pages: List[PageContent] = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text("text")

            images: List[str] = []
            if self.config.extract_images:
                images = self.extract_images_from_page(page, page_index)

            tables: List[Dict] = []
            if self.config.extract_tables and PDFPLUMBER_AVAILABLE:
                # Delegate table extraction to pdfplumber for accuracy
                tables = self._extract_tables_pdfplumber(pdf_path, page_index)

            pages.append(PageContent(
                page_num=page_index + 1,
                text=self.preprocess_text(text),
                tables=tables,
                images=images,
                metadata={"width": page.rect.width, "height": page.rect.height},
            ))

        doc.close()
        return ProcessedDocument(
            source_path=pdf_path,
            pages=pages,
            total_pages=len(pages),
            metadata={"parser": "pymupdf", "file": os.path.basename(pdf_path)},
        )

    # ------------------------------------------------------------------
    # Backend: pdfplumber
    # ------------------------------------------------------------------

    def parse_with_pdfplumber(self, pdf_path: str) -> ProcessedDocument:
        """Extract text and precise table data using pdfplumber."""
        if not PDFPLUMBER_AVAILABLE:
            raise RuntimeError("pdfplumber is not installed. Run: pip install pdfplumber")

        pages: List[PageContent] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = self.extract_tables_from_page(page, method="pdfplumber") if self.config.extract_tables else []
                images: List[str] = []
                if self.config.extract_images and PYMUPDF_AVAILABLE:
                    fitz_doc = fitz.open(pdf_path)
                    images = self.extract_images_from_page(fitz_doc[page_index], page_index)
                    fitz_doc.close()

                pages.append(PageContent(
                    page_num=page_index + 1,
                    text=self.preprocess_text(text),
                    tables=tables,
                    images=images,
                ))

        return ProcessedDocument(
            source_path=pdf_path,
            pages=pages,
            total_pages=len(pages),
            metadata={"parser": "pdfplumber", "file": os.path.basename(pdf_path)},
        )

    # ------------------------------------------------------------------
    # Backend: nougat (scientific PDFs)
    # ------------------------------------------------------------------

    def parse_with_nougat(self, pdf_path: str) -> ProcessedDocument:
        """
        Use the Nougat model for scientific PDFs that contain formulas,
        equations, and complex typesetting.  Falls back to pymupdf if
        nougat is not available.
        """
        if not NOUGAT_AVAILABLE:
            print("[PDFParser] nougat not installed, falling back to pymupdf. "
                  "Run: pip install nougat-ocr")
            return self.parse_with_pymupdf(pdf_path)

        # nougat produces one .mmd (Mathpix Markdown) file per PDF
        import subprocess
        output_dir = os.path.join(self.config.image_output_dir, "nougat_out")
        os.makedirs(output_dir, exist_ok=True)
        cmd = ["nougat", pdf_path, "--out", output_dir, "--no-skipping"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        mmd_files = list(Path(output_dir).glob("*.mmd"))
        if not mmd_files:
            print(f"[PDFParser] nougat produced no output for {pdf_path}, falling back.")
            return self.parse_with_pymupdf(pdf_path)

        mmd_text = mmd_files[0].read_text(encoding="utf-8")
        # Split by form-feed or double newline into pseudo-pages
        raw_pages = re.split(r"\n{3,}", mmd_text)
        pages = [
            PageContent(
                page_num=i + 1,
                text=self.preprocess_text(raw),
                metadata={"format": "mmd"},
            )
            for i, raw in enumerate(raw_pages) if raw.strip()
        ]

        return ProcessedDocument(
            source_path=pdf_path,
            pages=pages,
            total_pages=len(pages),
            metadata={"parser": "nougat", "file": os.path.basename(pdf_path)},
        )

    # ------------------------------------------------------------------
    # Backend: unstructured
    # ------------------------------------------------------------------

    def parse_with_unstructured(self, pdf_path: str) -> ProcessedDocument:
        """
        Use unstructured.io for mixed-layout documents (prose + tables + images).
        Groups elements by page into PageContent objects.
        """
        if not UNSTRUCTURED_AVAILABLE:
            print("[PDFParser] unstructured not installed, falling back to pymupdf. "
                  "Run: pip install unstructured")
            return self.parse_with_pymupdf(pdf_path)

        elements = partition_pdf(
            filename=pdf_path,
            extract_images_in_pdf=self.config.extract_images,
            infer_table_structure=self.config.extract_tables,
            strategy="hi_res",
        )

        # Group by page number
        page_map: Dict[int, List[str]] = {}
        page_tables: Dict[int, List[Dict]] = {}
        for el in elements:
            pn = getattr(el.metadata, "page_number", 1) or 1
            text_repr = str(el)
            page_map.setdefault(pn, []).append(text_repr)
            if el.category == "Table":
                page_tables.setdefault(pn, []).append({
                    "headers": [],
                    "rows": [],
                    "raw_text": text_repr,
                })

        pages = []
        for pn in sorted(page_map.keys()):
            pages.append(PageContent(
                page_num=pn,
                text=self.preprocess_text("\n".join(page_map[pn])),
                tables=page_tables.get(pn, []),
            ))

        return ProcessedDocument(
            source_path=pdf_path,
            pages=pages,
            total_pages=len(pages),
            metadata={"parser": "unstructured", "file": os.path.basename(pdf_path)},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def extract_images_from_page(self, page: Any, page_num: int) -> List[str]:
        """
        Extract embedded images from a PyMuPDF page and save them to disk.
        Returns a list of saved image file paths.
        """
        saved_paths: List[str] = []
        if not PYMUPDF_AVAILABLE:
            return saved_paths

        try:
            image_list = page.get_images(full=True)
            doc = page.parent  # fitz.Document reference
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                fname = f"page{page_num + 1}_img{img_index + 1}.{ext}"
                fpath = os.path.join(self.config.image_output_dir, fname)
                with open(fpath, "wb") as f:
                    f.write(img_bytes)
                saved_paths.append(fpath)
        except Exception as exc:
            print(f"[PDFParser] Image extraction error on page {page_num + 1}: {exc}")

        return saved_paths

    def extract_tables_from_page(self, page: Any, method: str = "pdfplumber") -> List[Dict]:
        """
        Extract tables from a pdfplumber page object.
        Returns a list of dicts with keys: headers, rows, raw_text.
        """
        tables: List[Dict] = []
        if method == "pdfplumber" and PDFPLUMBER_AVAILABLE:
            try:
                raw_tables = page.extract_tables()
                for raw in (raw_tables or []):
                    if not raw:
                        continue
                    headers = [str(h) if h else "" for h in raw[0]]
                    rows = [[str(c) if c else "" for c in row] for row in raw[1:]]
                    raw_text = "\n".join(["\t".join(headers)] + ["\t".join(r) for r in rows])
                    tables.append({"headers": headers, "rows": rows, "raw_text": raw_text})
            except Exception as exc:
                print(f"[PDFParser] Table extraction error: {exc}")
        return tables

    def _extract_tables_pdfplumber(self, pdf_path: str, page_index: int) -> List[Dict]:
        """Helper to extract tables from a specific page index via pdfplumber."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_index < len(pdf.pages):
                    return self.extract_tables_from_page(pdf.pages[page_index])
        except Exception:
            pass
        return []

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Normalize whitespace and strip common PDF extraction artifacts."""
        text = re.sub(r"\f", "\n", text)                 # form-feed → newline
        text = re.sub(r"[ \t]+", " ", text)              # collapse spaces/tabs
        text = re.sub(r"\n{3,}", "\n\n", text)           # max two consecutive newlines
        text = re.sub(r"-\n", "", text)                  # de-hyphenate line breaks
        return text.strip()

    def parse(self, pdf_path: str) -> ProcessedDocument:
        """
        Auto-select the best parser based on RAGConfig.pdf_model and
        library availability.
        """
        strategy = self.config.pdf_model.lower()

        if strategy == "nougat":
            return self.parse_with_nougat(pdf_path)
        elif strategy == "unstructured":
            return self.parse_with_unstructured(pdf_path)
        elif strategy == "pdfplumber":
            return self.parse_with_pdfplumber(pdf_path)
        elif strategy == "pymupdf":
            return self.parse_with_pymupdf(pdf_path)
        else:  # 'auto' — pick based on availability
            if PDFPLUMBER_AVAILABLE:
                return self.parse_with_pdfplumber(pdf_path)
            elif PYMUPDF_AVAILABLE:
                return self.parse_with_pymupdf(pdf_path)
            elif UNSTRUCTURED_AVAILABLE:
                return self.parse_with_unstructured(pdf_path)
            else:
                raise RuntimeError(
                    "No PDF parsing library available. Install one of: "
                    "pymupdf, pdfplumber, unstructured"
                )


# ---------------------------------------------------------------------------
# DocumentProcessor
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """
    High-level document loader and pre-processor.
    Delegates PDF parsing to PDFParser and handles plain text / Markdown files.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".rst", ".text"}

    def __init__(self, config: "RAGConfig"):
        self.config = config
        self.parser = PDFParser(config)

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_pdf(self, path: str) -> ProcessedDocument:
        """Parse a single PDF file and return a ProcessedDocument."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"PDF not found: {path}")
        return self.parser.parse(path)

    def load_text_file(self, path: str) -> ProcessedDocument:
        """Load a plain text or Markdown file as a single-page ProcessedDocument."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw_text = fh.read()

        text = self.preprocess_text(raw_text)
        page = PageContent(page_num=1, text=text, metadata={"file_type": Path(path).suffix})
        return ProcessedDocument(
            source_path=path,
            pages=[page],
            total_pages=1,
            metadata={"parser": "text", "file": os.path.basename(path)},
        )

    def load_directory(
        self,
        dir_path: str,
        extensions: Optional[List[str]] = None,
    ) -> List[ProcessedDocument]:
        """
        Recursively load all supported documents from a directory.

        Parameters
        ----------
        dir_path   : Path to the directory to scan.
        extensions : File extensions to include (default: pdf, txt, md).
        """
        if extensions is None:
            extensions = [".pdf", ".txt", ".md"]
        extensions = {ext.lower() for ext in extensions}
        documents: List[ProcessedDocument] = []

        for root, _dirs, files in os.walk(dir_path):
            for fname in sorted(files):
                ext = Path(fname).suffix.lower()
                if ext not in extensions:
                    continue
                full_path = os.path.join(root, fname)
                try:
                    doc = self.process(full_path)
                    documents.extend(doc)
                    print(f"[DocumentProcessor] Loaded: {full_path}")
                except Exception as exc:
                    print(f"[DocumentProcessor] Failed to load {full_path}: {exc}")

        return documents

    def process(self, source: str) -> List[ProcessedDocument]:
        """
        Unified entry point.  Accepts a file path or directory path
        and returns a list of ProcessedDocument objects.
        """
        if os.path.isdir(source):
            return self.load_directory(source)

        ext = Path(source).suffix.lower()
        if ext == ".pdf":
            return [self.load_pdf(source)]
        elif ext in {".txt", ".md", ".rst", ".text"}:
            return [self.load_text_file(source)]
        else:
            print(f"[DocumentProcessor] Unsupported extension '{ext}' for {source}. "
                  "Attempting text load.")
            return [self.load_text_file(source)]

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Clean and normalize text: collapse whitespace, strip artifacts."""
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
