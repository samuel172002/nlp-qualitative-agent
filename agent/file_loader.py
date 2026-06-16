from __future__ import annotations
import os
import re
import uuid
from pathlib import Path

from shared.models import TextSegment

CHUNK_SIZE = 2500   # target characters per segment
OVERLAP = 200       # character overlap between chunks


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def _split_into_chunks(text: str, source_file: str) -> list[TextSegment]:
    """Split text into ~CHUNK_SIZE character segments on sentence boundaries."""
    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Simple sentence splitter: split on '. ', '? ', '! ', or '\n\n'
    sentence_pattern = re.compile(r'(?<=[.!?])\s+|(?<=\n)\n')
    sentences = sentence_pattern.split(text)
    # Fallback: keep non-empty
    sentences = [s.strip() for s in sentences if s.strip()]

    segments: list[TextSegment] = []
    current = ""
    seg_index = 0

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= CHUNK_SIZE:
            current = (current + " " + sentence).strip()
        else:
            if current:
                seg_id = f"{Path(source_file).stem}_seg{seg_index:04d}_{uuid.uuid4().hex[:6]}"
                segments.append(TextSegment(text=current, segment_id=seg_id, source_file=source_file))
                seg_index += 1
                # Carry overlap: keep last OVERLAP chars from previous chunk
                current = current[-OVERLAP:] + " " + sentence
                current = current.strip()
            else:
                current = sentence

    if current:
        seg_id = f"{Path(source_file).stem}_seg{seg_index:04d}_{uuid.uuid4().hex[:6]}"
        segments.append(TextSegment(text=current, segment_id=seg_id, source_file=source_file))

    return segments


class FileLoader:
    READERS = {
        ".txt": _read_txt,
        ".pdf": _read_pdf,
        ".docx": _read_docx,
    }

    def load(self, file_paths: list[str]) -> list[TextSegment]:
        segments: list[TextSegment] = []
        for path in file_paths:
            ext = Path(path).suffix.lower()
            reader = self.READERS.get(ext)
            if reader is None:
                raise ValueError(f"Unsupported file type: {ext}")
            text = reader(path)
            chunks = _split_into_chunks(text, path)
            segments.extend(chunks)
        return segments
