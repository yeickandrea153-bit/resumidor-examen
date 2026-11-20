"""Extractor v2.0 para DOCX, PDF (2 columnas + filtro cabeceras), PPTX (viñetas) y OCR con filtro de calidad."""
import io
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Set, Tuple

from core.nlp_utils import (
    clean_text,
    is_chapter_heading,
    is_clean_ocr_text,
    is_excluded_section,
    is_toc_or_bibliography_line,
    looks_like_code,
    refold_paragraphs,
)


@dataclass
class ExtractedSection:
    title: str
    paragraphs: List[str] = field(default_factory=list)


@dataclass
class ExtractedImageText:
    source_location: str
    text: str
    is_code: bool


@dataclass
class ExtractionResult:
    filename: str
    file_type: str
    sections: List[ExtractedSection] = field(default_factory=list)
    code_blocks: List[Tuple[str, str]] = field(default_factory=list)
    image_ocr_results: List[ExtractedImageText] = field(default_factory=list)
    total_images_found: int = 0
    removed_headers_footers: List[str] = field(default_factory=list)
    excluded_sections: List[str] = field(default_factory=list)
    ocr_available: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def full_theory_text(self) -> str:
        chunks = []
        for sec in self.sections:
            for p in sec.paragraphs:
                chunks.append(p)
        return "\n\n".join(chunks)

    @property
    def headings(self) -> List[str]:
        return [s.title for s in self.sections if s.title and s.title != "Contenido General"]


def check_ocr_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def run_ocr_on_bytes(image_bytes: bytes, lang: str = "spa+eng") -> str:
    """Ejecuta OCR con preprocesamiento y filtro de calidad contra basura visual."""
    try:
        from PIL import Image, ImageOps, ImageFilter
        import pytesseract

        img = Image.open(io.BytesIO(image_bytes))
        if img.width < 60 or img.height < 30:
            return ""

        gray = ImageOps.grayscale(img)
        if gray.width < 800:
            gray = gray.resize((gray.width * 2, gray.height * 2))
        gray = ImageOps.autocontrast(gray)
        gray = gray.filter(ImageFilter.SHARPEN)

        try:
            text = pytesseract.image_to_string(gray, lang=lang)
        except Exception:
            text = pytesseract.image_to_string(gray)

        cleaned = clean_text(text)
        if not is_clean_ocr_text(cleaned):
            return ""
        return cleaned
    except Exception:
        return ""


def _add_paragraph_or_code(
    text: str,
    current_section: ExtractedSection,
    code_blocks: List[Tuple[str, str]],
    location: str,
):
    cleaned = clean_text(text)
    if not cleaned:
        return
    if looks_like_code(cleaned):
        code_blocks.append((location, cleaned))
    else:
        rebuilt = refold_paragraphs(cleaned)
        if rebuilt:
            current_section.paragraphs.append(rebuilt)


def _filter_excluded_sections(result: ExtractionResult):
    """Elimina secciones de Índice, Referencias, Bibliografía o Anexos para no contaminar el examen."""
    kept: List[ExtractedSection] = []
    for sec in result.sections:
        if is_excluded_section(sec.title):
            result.excluded_sections.append(sec.title)
            continue
        if sec.paragraphs:
            kept.append(sec)
    result.sections = kept if kept else result.sections


def _detect_repeated_headers_footers(pages_lines: List[List[str]]) -> Set[str]:
    """Detecta cabeceras, pies de página y marcas de agua repetidas en >= 35% de las páginas."""
    num_pages = len(pages_lines)
    if num_pages < 3:
        return set()

    line_page_counts: Counter = Counter()
    for lines in pages_lines:
        # Revisar primeras 3 líneas, últimas 3 líneas y cualquier línea corta repetida (marca de agua)
        candidate_lines = set()
        for idx, l in enumerate(lines):
            norm = re.sub(r"\d+", "#", l.strip().lower())
            norm = re.sub(r"\s+", " ", norm)
            if not norm or len(norm) > 120:
                continue
            if idx < 3 or idx >= len(lines) - 3 or len(norm) < 60:
                candidate_lines.add(norm)
        for c in candidate_lines:
            line_page_counts[c] += 1

    threshold = max(3, int(num_pages * 0.35))
    return {line for line, cnt in line_page_counts.items() if cnt >= threshold}


def _extract_pdf_page_text_columns(page) -> str:
    """Extrae texto de una página PDF respetando diseños de 1 o 2 columnas mediante coordenadas."""
    blocks = page.get_text("blocks")
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    if not text_blocks:
        return ""

    page_width = page.rect.width
    mid_x = page_width / 2.0

    # Verificar si hay bloques claramente confinados a columna izquierda y columna derecha
    left_col = [b for b in text_blocks if b[2] <= mid_x + 15]
    right_col = [b for b in text_blocks if b[0] >= mid_x - 15]
    spanning = [b for b in text_blocks if b not in left_col and b not in right_col]

    if len(left_col) >= 2 and len(right_col) >= 2:
        # Diseño a 2 columnas: encabezados anchos arriba, luego col izquierda, luego col derecha
        top_span = sorted([b for b in spanning if b[1] < min(x[1] for x in left_col + right_col)], key=lambda b: b[1])
        bottom_span = sorted([b for b in spanning if b not in top_span], key=lambda b: b[1])
        ordered = (
            top_span
            + sorted(left_col, key=lambda b: (b[1], b[0]))
            + sorted(right_col, key=lambda b: (b[1], b[0]))
            + bottom_span
        )
    else:
        ordered = sorted(text_blocks, key=lambda b: (round(b[1], 1), b[0]))

    return "\n\n".join(b[4].strip() for b in ordered)


def extract_from_docx(file_bytes: bytes, filename: str, use_ocr: bool = True) -> ExtractionResult:
    import docx

    result = ExtractionResult(
        filename=filename,
        file_type="DOCX",
        ocr_available=check_ocr_available(),
    )
    document = docx.Document(io.BytesIO(file_bytes))
    current_section = ExtractedSection(title="Contenido General")
    result.sections.append(current_section)

    for idx, para in enumerate(document.paragraphs):
        txt = para.text.strip()
        if not txt or is_toc_or_bibliography_line(txt):
            continue
        style_name = (para.style.name or "").lower() if para.style else ""
        is_heading_style = (
            "heading" in style_name
            or "título" in style_name
            or "titulo" in style_name
            or style_name == "title"
        )
        if is_heading_style or is_chapter_heading(txt):
            if current_section.paragraphs or current_section.title != "Contenido General":
                current_section = ExtractedSection(title=txt)
                result.sections.append(current_section)
            else:
                current_section.title = txt
        else:
            _add_paragraph_or_code(txt, current_section, result.code_blocks, f"Párrafo {idx + 1}")

    for table in document.tables:
        rows_text = []
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells if c.text.strip()]
            if cells:
                rows_text.append(" | ".join(cells))
        if rows_text:
            current_section.paragraphs.append(". ".join(rows_text) + ".")

    image_rels = [rel for rel in document.part.rels.values() if "image" in rel.target_ref]
    result.total_images_found = len(image_rels)

    if use_ocr and image_rels:
        if not result.ocr_available:
            result.warnings.append(
                f"Se detectaron {len(image_rels)} imagen(es), pero Tesseract OCR no está instalado en este entorno."
            )
        else:
            for i, rel in enumerate(image_rels, start=1):
                try:
                    ocr_text = run_ocr_on_bytes(rel.target_part.blob)
                    if ocr_text:
                        is_code = looks_like_code(ocr_text)
                        loc = f"Imagen {i} (Word)"
                        result.image_ocr_results.append(ExtractedImageText(loc, ocr_text, is_code))
                        if is_code:
                            result.code_blocks.append((loc, ocr_text))
                        else:
                            current_section.paragraphs.append(refold_paragraphs(ocr_text))
                except Exception:
                    continue

    _filter_excluded_sections(result)
    return result


def extract_from_pdf(file_bytes: bytes, filename: str, use_ocr: bool = True) -> ExtractionResult:
    import fitz  # PyMuPDF

    result = ExtractionResult(
        filename=filename,
        file_type="PDF",
        ocr_available=check_ocr_available(),
    )
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    # Paso 1: Extraer texto crudo de cada página respetando columnas para detectar cabeceras repetidas
    raw_pages_text: List[str] = []
    pages_lines: List[List[str]] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        p_text = clean_text(_extract_pdf_page_text_columns(page))
        raw_pages_text.append(p_text)
        pages_lines.append([l.strip() for l in p_text.splitlines() if l.strip()])

    repeated_patterns = _detect_repeated_headers_footers(pages_lines)
    if repeated_patterns:
        result.removed_headers_footers = sorted(list(repeated_patterns))[:8]

    current_section = ExtractedSection(title="Contenido General")
    result.sections.append(current_section)
    seen_xrefs = set()

    # Paso 2: Limpiar cabeceras/pies, reconstruir párrafos y detectar capítulos
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_label = f"Página {page_num + 1}"
        p_text = raw_pages_text[page_num]

        # Si la página casi no tiene texto, intentar OCR de página completa (escaneado)
        if len(p_text) < 30 and use_ocr and result.ocr_available:
            pix = page.get_pixmap(dpi=200)
            ocr_page_text = run_ocr_on_bytes(pix.tobytes("png"))
            if ocr_page_text:
                is_code = looks_like_code(ocr_page_text)
                result.image_ocr_results.append(
                    ExtractedImageText(f"{page_label} (Escaneada)", ocr_page_text, is_code)
                )
                if is_code:
                    result.code_blocks.append((f"{page_label} (Escaneada)", ocr_page_text))
                else:
                    current_section.paragraphs.append(refold_paragraphs(ocr_page_text))
        else:
            filtered_lines: List[str] = []
            for raw_line in p_text.splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    filtered_lines.append("")
                    continue
                norm = re.sub(r"\d+", "#", stripped.lower())
                norm = re.sub(r"\s+", " ", norm)
                if norm in repeated_patterns or is_toc_or_bibliography_line(stripped):
                    continue
                filtered_lines.append(stripped)

            cleaned_page = refold_paragraphs("\n".join(filtered_lines))
            for block in cleaned_page.split("\n\n"):
                block = block.strip()
                if not block:
                    continue
                if is_chapter_heading(block):
                    title_clean = block.title() if block.isupper() else block
                    if current_section.paragraphs or current_section.title != "Contenido General":
                        current_section = ExtractedSection(title=title_clean)
                        result.sections.append(current_section)
                    else:
                        current_section.title = title_clean
                else:
                    _add_paragraph_or_code(block, current_section, result.code_blocks, page_label)

        # Extraer imágenes incrustadas
        for img_idx, img_info in enumerate(page.get_images(full=True), start=1):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            result.total_images_found += 1

            if use_ocr and result.ocr_available:
                try:
                    base_img = doc.extract_image(xref)
                    ocr_text = run_ocr_on_bytes(base_img["image"])
                    if ocr_text:
                        is_code = looks_like_code(ocr_text)
                        loc = f"{page_label} - Imagen {img_idx}"
                        result.image_ocr_results.append(ExtractedImageText(loc, ocr_text, is_code))
                        if is_code:
                            result.code_blocks.append((loc, ocr_text))
                        else:
                            current_section.paragraphs.append(refold_paragraphs(ocr_text))
                except Exception:
                    continue

    if use_ocr and result.total_images_found > 0 and not result.ocr_available:
        result.warnings.append(
            f"Se detectaron {result.total_images_found} imagen(es) en el PDF, pero Tesseract OCR no está instalado."
        )

    doc.close()
    _filter_excluded_sections(result)
    return result


def extract_from_pptx(file_bytes: bytes, filename: str, use_ocr: bool = True) -> ExtractionResult:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    result = ExtractionResult(
        filename=filename,
        file_type="PPTX",
        ocr_available=check_ocr_available(),
    )
    prs = Presentation(io.BytesIO(file_bytes))

    for slide_idx, slide in enumerate(prs.slides, start=1):
        slide_label = f"Diapositiva {slide_idx}"
        slide_title = slide_label
        if slide.shapes.title and slide.shapes.title.text.strip():
            slide_title = f"{slide_label}: {slide.shapes.title.text.strip()}"

        section = ExtractedSection(title=slide_title)
        result.sections.append(section)

        img_counter = 0
        for shape in slide.shapes:
            if shape == slide.shapes.title:
                continue

            if shape.has_text_frame:
                raw_lines = [p.text.strip() for p in shape.text_frame.paragraphs if p.text.strip()]
                if raw_lines:
                    joined_raw = "\n".join(raw_lines)
                    if looks_like_code(joined_raw):
                        result.code_blocks.append((slide_label, clean_text(joined_raw)))
                    else:
                        # Agregar punto final a viñetas que no tengan puntuación de cierre
                        normalized_bullets = []
                        for ln in raw_lines:
                            ln_clean = re.sub(r"^[•●▪◦\-–—*]\s*", "", ln).strip()
                            if len(ln_clean) < 12:
                                continue
                            if not re.search(r"[.!?]$", ln_clean):
                                ln_clean += "."
                            normalized_bullets.append(ln_clean)
                        if normalized_bullets:
                            section.paragraphs.append(" ".join(normalized_bullets))

            if shape.has_table:
                rows_text = []
                for row in shape.table.rows:
                    cells = [c.text.strip().replace("\n", " ") for c in row.cells if c.text.strip()]
                    if cells:
                        rows_text.append(" | ".join(cells))
                if rows_text:
                    section.paragraphs.append(". ".join(rows_text) + ".")

            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                img_counter += 1
                result.total_images_found += 1
                if use_ocr and result.ocr_available:
                    try:
                        ocr_text = run_ocr_on_bytes(shape.image.blob)
                        if ocr_text:
                            is_code = looks_like_code(ocr_text)
                            loc = f"{slide_label} - Imagen {img_counter}"
                            result.image_ocr_results.append(ExtractedImageText(loc, ocr_text, is_code))
                            if is_code:
                                result.code_blocks.append((loc, ocr_text))
                            else:
                                section.paragraphs.append(refold_paragraphs(ocr_text))
                    except Exception:
                        continue

        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                _add_paragraph_or_code(notes, section, result.code_blocks, f"{slide_label} (Notas)")

    if use_ocr and result.total_images_found > 0 and not result.ocr_available:
        result.warnings.append(
            f"Se detectaron {result.total_images_found} imagen(es) en el PPTX, pero Tesseract OCR no está instalado."
        )

    _filter_excluded_sections(result)
    return result


def extract_from_raw_text(raw_text: str) -> ExtractionResult:
    result = ExtractionResult(filename="Texto pegado", file_type="TXT", ocr_available=False)
    current_section = ExtractedSection(title="Texto Ingresado")
    result.sections.append(current_section)

    rebuilt = refold_paragraphs(raw_text)
    for block in rebuilt.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if is_chapter_heading(block):
            if current_section.paragraphs or current_section.title != "Texto Ingresado":
                current_section = ExtractedSection(title=block)
                result.sections.append(current_section)
            else:
                current_section.title = block
        else:
            _add_paragraph_or_code(block, current_section, result.code_blocks, "Texto pegado")

    _filter_excluded_sections(result)
    return result


def extract_document(file_bytes: bytes, filename: str, use_ocr: bool = True) -> ExtractionResult:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".docx":
        return extract_from_docx(file_bytes, filename, use_ocr=use_ocr)
    elif ext == ".pdf":
        return extract_from_pdf(file_bytes, filename, use_ocr=use_ocr)
    elif ext == ".pptx":
        return extract_from_pptx(file_bytes, filename, use_ocr=use_ocr)
    elif ext == ".txt":
        return extract_from_raw_text(file_bytes.decode("utf-8", errors="ignore"))
    else:
        raise ValueError(f"Formato no soportado: {ext}. Usa .docx, .pdf, .pptx o .txt")
