"""Exportación de la guía de estudio (Resumen + Glosario + Examen) a DOCX y Markdown."""
import io
from typing import List

import docx
from docx.shared import Pt

from core.extractor import ExtractionResult
from core.preguntas import ExamQuestion
from core.resumidor import SummaryResult

TYPE_LABELS = {
    "opcion_multiple": "Opción Múltiple",
    "completar": "Completar espacios",
    "desarrollo": "Pregunta de Desarrollo",
    "verdadero_falso": "Verdadero / Falso",
}


def export_to_markdown(
    extraction: ExtractionResult,
    summary: SummaryResult,
    questions: List[ExamQuestion],
) -> str:
    lines = [
        f"# Guía de Estudio: {extraction.filename}",
        "",
        "## 1. Resumen Estructurado",
        "",
    ]

    for sec_title, sents in summary.by_section.items():
        lines.append(f"### {sec_title}")
        for s in sents:
            lines.append(f"- {s}")
        lines.append("")

    if summary.glossary:
        lines.append("## 2. Conceptos Clave / Glosario")
        lines.append("")
        for term, definition in summary.glossary:
            lines.append(f"- **{term}:** {definition}")
        lines.append("")

    if questions:
        lines.append("## 3. Posibles Preguntas de Examen")
        lines.append("")
        for idx, q in enumerate(questions, start=1):
            label = TYPE_LABELS.get(q.q_type, q.q_type)
            lines.append(f"**{idx}. [{label}]** {q.question}")
            if q.options:
                for opt_i, opt in enumerate(q.options):
                    letter = chr(65 + opt_i)
                    lines.append(f"   - {letter}) {opt}")
            lines.append("")

        lines.append("---")
        lines.append("## 4. Solucionario")
        lines.append("")
        for idx, q in enumerate(questions, start=1):
            lines.append(f"**{idx}. Respuesta:** {q.answer}")
            if q.explanation:
                lines.append(f"   - *Nota:* {q.explanation}")
        lines.append("")

    if extraction.code_blocks:
        lines.append("## 5. Bloques de Código / Capturas Detectadas")
        lines.append("")
        for loc, code in extraction.code_blocks:
            lines.append(f"### {loc}")
            lines.append("```")
            lines.append(code)
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def export_to_docx_bytes(
    extraction: ExtractionResult,
    summary: SummaryResult,
    questions: List[ExamQuestion],
) -> bytes:
    doc = docx.Document()
    doc.add_heading(f"Guía de Estudio: {extraction.filename}", level=0)

    doc.add_heading("1. Resumen Estructurado", level=1)
    for sec_title, sents in summary.by_section.items():
        doc.add_heading(sec_title, level=2)
        for s in sents:
            doc.add_paragraph(s, style="List Bullet")

    if summary.glossary:
        doc.add_heading("2. Conceptos Clave / Glosario", level=1)
        for term, definition in summary.glossary:
            p = doc.add_paragraph()
            run_term = p.add_run(f"{term}: ")
            run_term.bold = True
            p.add_run(definition)

    if questions:
        doc.add_heading("3. Posibles Preguntas de Examen", level=1)
        for idx, q in enumerate(questions, start=1):
            label = TYPE_LABELS.get(q.q_type, q.q_type)
            p = doc.add_paragraph()
            r_num = p.add_run(f"{idx}. [{label}] ")
            r_num.bold = True
            p.add_run(q.question)

            if q.options:
                for opt_i, opt in enumerate(q.options):
                    letter = chr(65 + opt_i)
                    doc.add_paragraph(f"    {letter}) {opt}")

        doc.add_page_break()
        doc.add_heading("4. Solucionario", level=1)
        for idx, q in enumerate(questions, start=1):
            p = doc.add_paragraph()
            r_num = p.add_run(f"{idx}. Respuesta: ")
            r_num.bold = True
            p.add_run(q.answer)
            if q.explanation:
                p_exp = doc.add_paragraph(f"Nota: {q.explanation}")
                for run in p_exp.runs:
                    run.italic = True
                    run.font.size = Pt(9.5)

    if extraction.code_blocks:
        doc.add_heading("5. Bloques de Código Detectados", level=1)
        for loc, code in extraction.code_blocks:
            doc.add_heading(loc, level=2)
            p_code = doc.add_paragraph(code)
            for run in p_code.runs:
                run.font.name = "Consolas"
                run.font.size = Pt(9)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()
