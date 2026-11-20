"""Aplicación Streamlit v2.0: Resumidor de Documentos y Generador de Exámenes sin IA."""
import streamlit as st

from core.exportador import TYPE_LABELS, export_to_docx_bytes, export_to_markdown
from core.extractor import check_ocr_available, extract_document, extract_from_raw_text
from core.nlp_utils import load_spacy_model
from core.preguntas import generate_exam_questions
from core.resumidor import summarize_document

st.set_page_config(
    page_title="EstudiaSinIA v2 — Resumidor y Generador de Exámenes",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource(show_spinner="Cargando motor lingüístico en español...")
def get_nlp():
    return load_spacy_model()


def main():
    st.title("📚 Resumidor de Documentos y Generador de Exámenes")
    st.caption(
        "v2.0 — 100% sin APIs externas: TextRank por capítulos, reconstrucción de párrafos, filtro de cabeceras/índices y OCR."
    )

    nlp = get_nlp()
    ocr_ready = check_ocr_available()

    # --- Barra lateral ---
    with st.sidebar:
        st.header("⚙️ Configuración")

        ratio = st.slider(
            "Densidad del resumen (%)",
            min_value=10,
            max_value=50,
            value=25,
            step=5,
            help="Porcentaje aproximado de oraciones clave por cada sección o capítulo.",
        ) / 100.0

        auto_proportional = st.toggle(
            "Extensión proporcional automática",
            value=True,
            help="Adapta el largo del resumen automáticamente según si subes un apunte corto o un libro de 150 páginas.",
        )

        max_sentences = None
        if not auto_proportional:
            max_sentences = st.slider(
                "Límite manual de oraciones",
                min_value=5,
                max_value=80,
                value=20,
            )

        st.divider()
        st.subheader("📝 Generador de Examen")
        num_questions = st.slider("Cantidad de preguntas", min_value=3, max_value=30, value=10)

        q_types = st.multiselect(
            "Tipos de preguntas",
            options=["opcion_multiple", "completar", "desarrollo", "verdadero_falso"],
            default=["opcion_multiple", "completar", "desarrollo", "verdadero_falso"],
            format_func=lambda x: TYPE_LABELS.get(x, x),
        )

        seed = st.number_input("Semilla aleatoria (cámbiala para variar preguntas)", min_value=1, max_value=9999, value=42)

        st.divider()
        st.subheader("🖼️ Lectura de Imágenes (OCR)")
        use_ocr = st.toggle(
            "Extraer texto y código de imágenes",
            value=ocr_ready,
            disabled=not ocr_ready,
            help="Usa Tesseract OCR con filtro de calidad para leer capturas de código y etiquetas de diagramas.",
        )

        st.divider()
        st.markdown("**Estado del motor local:**")
        st.write(f"- spaCy (español): {'✅ Activo' if nlp else '⚠️ Modo respaldo (regex)'}")
        st.write(f"- Tesseract OCR: {'✅ Activo' if ocr_ready else '⚠️ No instalado'}")

    # --- Entrada ---
    input_mode = st.radio(
        "¿Cómo deseas cargar tu material de estudio?",
        ["Subir archivo (Word, PDF, PowerPoint, TXT)", "Pegar texto directamente"],
        horizontal=True,
    )

    extraction = None

    if input_mode == "Subir archivo (Word, PDF, PowerPoint, TXT)":
        uploaded_file = st.file_uploader(
            "Arrastra o selecciona tu documento",
            type=["docx", "pdf", "pptx", "txt"],
            help="Soporta .docx, .pdf (1 o 2 columnas), .pptx y .txt",
        )
        if uploaded_file is not None:
            with st.spinner("Extrayendo texto, limpiando cabeceras/índices y reconstruyendo párrafos..."):
                extraction = extract_document(
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                    use_ocr=use_ocr,
                )
    else:
        raw_text = st.text_area(
            "Pega aquí tus apuntes o el contenido del tema:",
            height=220,
            placeholder="Pega al menos 2 o 3 párrafos para generar un resumen y preguntas útiles...",
        )
        if st.button("Analizar texto", type="primary") or (raw_text and len(raw_text.strip()) > 100):
            if raw_text.strip():
                extraction = extract_from_raw_text(raw_text)

    if extraction is None:
        st.info("👆 Sube un archivo `.docx`, `.pdf`, `.pptx` o pega texto para comenzar.")
        return

    for warn in extraction.warnings:
        st.warning(warn)

    # Mostrar limpieza automática realizada (transparencia para el usuario)
    cleanup_notes = []
    if extraction.removed_headers_footers:
        cleanup_notes.append(
            f"Se filtraron **{len(extraction.removed_headers_footers)}** cabecera(s)/pie(s) repetidos automáticamente."
        )
    if extraction.excluded_sections:
        cleanup_notes.append(
            f"Se excluyeron secciones no resumibles: *{', '.join(extraction.excluded_sections)}*."
        )
    if cleanup_notes:
        st.caption("🧹 Limpieza automática: " + " | ".join(cleanup_notes))

    if not extraction.full_theory_text.strip() and not extraction.code_blocks:
        st.error("No se pudo extraer texto legible del documento. Si es un PDF escaneado, asegúrate de tener OCR activo.")
        return

    # --- Procesamiento NLP ---
    with st.spinner("Resumiendo por capítulos (TextRank + Heurísticas + MMR) y generando examen..."):
        summary = summarize_document(
            extraction,
            ratio=ratio,
            max_sentences=max_sentences,
            nlp=nlp,
        )
        questions = generate_exam_questions(
            summary,
            num_questions=num_questions,
            selected_types=q_types,
            nlp=nlp,
            seed=int(seed),
        )

    # --- Métricas ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Oraciones válidas", summary.total_sentences)
    c2.metric("Resumen / Secciones", f"{len(summary.selected_sentences)} ({summary.total_sections} sec.)")
    c3.metric("Preguntas generadas", len(questions))
    c4.metric("Bloques código / OCR", len(extraction.code_blocks))

    # --- Pestañas ---
    tab_resumen, tab_examen, tab_quiz, tab_glosario, tab_codigo, tab_export = st.tabs([
        "📄 Resumen",
        "📝 Banco de Preguntas",
        "🎯 Modo Simulacro",
        "🔑 Conceptos Clave",
        "💻 Código e Imágenes (OCR)",
        "⬇️ Exportar",
    ])

    with tab_resumen:
        st.subheader("Resumen estructurado por secciones / capítulos")
        if not summary.by_section:
            st.warning("El texto teórico es demasiado corto para resumir.")
        else:
            for sec_title, sents in summary.by_section.items():
                st.markdown(f"### {sec_title}")
                for s in sents:
                    st.markdown(f"- {s}")

    with tab_examen:
        st.subheader("Posibles preguntas de examen")
        show_answers = st.toggle("Mostrar respuestas y explicaciones", value=False)
        if not questions:
            st.info("No se pudieron generar preguntas con los filtros actuales. Prueba subir un texto más extenso.")
        for idx, q in enumerate(questions, start=1):
            label = TYPE_LABELS.get(q.q_type, q.q_type)
            with st.container(border=True):
                st.markdown(f"**{idx}. [{label}]** *(Sección: {q.source_section})*")
                st.write(q.question)
                if q.options:
                    for opt_i, opt in enumerate(q.options):
                        st.write(f"   **{chr(65 + opt_i)})** {opt}")
                if show_answers:
                    st.success(f"**Respuesta:** {q.answer}")
                    if q.explanation:
                        st.caption(q.explanation)

    with tab_quiz:
        st.subheader("Simulacro interactivo")
        interactive_qs = [q for q in questions if q.q_type in ("opcion_multiple", "verdadero_falso")]
        if not interactive_qs:
            st.info("Activa 'Opción Múltiple' o 'Verdadero / Falso' en la barra lateral para usar el simulacro.")
        else:
            user_answers = {}
            for idx, q in enumerate(interactive_qs, start=1):
                st.markdown(f"**{idx}.** {q.question}")
                user_answers[idx] = st.radio(
                    f"Selecciona tu respuesta para la pregunta {idx}",
                    options=q.options,
                    index=None,
                    key=f"quiz_q_{idx}_{seed}",
                    label_visibility="collapsed",
                )
                st.divider()

            if st.button("Calificar simulacro", type="primary"):
                correct_count = 0
                for idx, q in enumerate(interactive_qs, start=1):
                    ans = user_answers.get(idx)
                    if ans == q.answer:
                        correct_count += 1
                        st.success(f"Pregunta {idx}: ¡Correcto! ({q.answer})")
                    else:
                        st.error(f"Pregunta {idx}: Incorrecto. Tu respuesta: {ans or 'Sin responder'} | Correcta: {q.answer}")
                        if q.explanation:
                            st.caption(q.explanation)
                score_pct = (correct_count / len(interactive_qs)) * 100
                st.metric("Puntaje final", f"{correct_count} / {len(interactive_qs)} ({score_pct:.0f}%)")

    with tab_glosario:
        st.subheader("Palabras clave detectadas (TF-IDF)")
        if summary.keywords:
            st.write(", ".join([f"`{k}`" for k, _ in summary.keywords]))
        st.subheader("Mini-glosario automático")
        if summary.glossary:
            for term, definition in summary.glossary:
                with st.expander(f"📌 {term}", expanded=True):
                    st.write(definition)
        else:
            st.info("No se detectaron definiciones suficientes para armar el glosario.")

    with tab_codigo:
        st.subheader("Bloques de código detectados (listos para copiar)")
        st.caption("Se separan automáticamente del texto teórico para que no contaminen el resumen.")
        if extraction.code_blocks:
            for loc, code in extraction.code_blocks:
                st.markdown(f"**{loc}**")
                st.code(code)
        else:
            st.info("No se detectaron bloques de código en el documento.")

        if extraction.image_ocr_results:
            st.subheader("Todo el texto extraído de imágenes (OCR)")
            for item in extraction.image_ocr_results:
                tag = "Código" if item.is_code else "Texto / Etiquetas"
                with st.expander(f"🖼️ {item.source_location} — [{tag}]"):
                    if item.is_code:
                        st.code(item.text)
                    else:
                        st.write(item.text)

    with tab_export:
        st.subheader("Descargar guía de estudio")
        md_data = export_to_markdown(extraction, summary, questions)
        docx_data = export_to_docx_bytes(extraction, summary, questions)

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                label="📄 Descargar en Word (.docx)",
                data=docx_data,
                file_name=f"guia_estudio_{extraction.filename.rsplit('.', 1)[0]}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        with col_b:
            st.download_button(
                label="📝 Descargar en Markdown (.md)",
                data=md_data.encode("utf-8"),
                file_name=f"guia_estudio_{extraction.filename.rsplit('.', 1)[0]}.md",
                mime="text/markdown",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
