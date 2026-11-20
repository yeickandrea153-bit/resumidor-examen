"""Generador de preguntas de examen v2.0 (Opción Múltiple, Cloze, Desarrollo académico/narrativo y V/F)."""
import random
import re
from dataclasses import dataclass, field
from typing import List, Optional

from core.nlp_utils import SPANISH_STOPWORDS, normalize_word, tokenize_words
from core.resumidor import SummaryResult


@dataclass
class ExamQuestion:
    q_type: str  # "opcion_multiple" | "completar" | "desarrollo" | "verdadero_falso"
    question: str
    answer: str
    options: List[str] = field(default_factory=list)
    explanation: str = ""
    source_section: str = ""


def _extract_candidate_terms(sentence: str, keywords: List[str], nlp=None) -> List[str]:
    candidates: List[str] = []

    for kw in keywords:
        if len(kw) >= 4:
            match = re.search(rf"\b{re.escape(kw)}\b", sentence, re.IGNORECASE)
            if match:
                candidates.append(match.group(0))

    if nlp is not None:
        doc = nlp(sentence)
        for ent in doc.ents:
            if len(ent.text.strip()) >= 3:
                candidates.append(ent.text.strip())
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and len(token.text) >= 5 and token.text.lower() not in SPANISH_STOPWORDS:
                candidates.append(token.text)
    else:
        mid_caps = re.findall(r"(?<!^)\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{3,}\b", sentence)
        candidates.extend(mid_caps)
        long_words = [w for w in tokenize_words(sentence) if len(w) >= 6]
        candidates.extend(long_words)

    unique = []
    seen = set()
    for c in candidates:
        cn = normalize_word(c)
        if cn not in seen and len(cn) >= 4:
            seen.add(cn)
            unique.append(c)
    return unique


def _build_distractors(correct_term: str, pool: List[str], n: int = 3) -> List[str]:
    correct_norm = normalize_word(correct_term)
    correct_is_multi = " " in correct_term.strip()

    filtered = []
    seen = {correct_norm}
    for item in pool:
        item_clean = item.strip()
        norm = normalize_word(item_clean)
        if not norm or norm in seen:
            continue
        if correct_norm in norm or norm in correct_norm:
            continue
        seen.add(norm)
        filtered.append(item_clean)

    same_shape = [x for x in filtered if (" " in x) == correct_is_multi]
    other_shape = [x for x in filtered if (" " in x) != correct_is_multi]
    ordered = same_shape + other_shape

    if len(ordered) <= n:
        return ordered
    return random.sample(ordered[: max(n * 3, len(ordered))], n)


def _generate_open_question(sentence: str, section: str, nlp=None) -> Optional[ExamQuestion]:
    """Detecta patrones académicos (definición, función, clasificación, causa) o entidades narrativas."""
    patterns = [
        (
            r"^([A-ZÁÉÍÓÚÑ][^,.;:]{3,65}?)\s+(?:se define como|consiste en|es\s+(?:un|una|el|la))\s+(.{15,})",
            lambda m: f"¿Qué es o en qué consiste «{m.group(1).strip()}»?",
        ),
        (
            r"^([A-ZÁÉÍÓÚÑ][^,.;:]{3,65}?)\s+(?:permite|sirve para|tiene como objetivo|tiene como finalidad)\s+(.{15,})",
            lambda m: f"¿Cuál es la función u objetivo principal de «{m.group(1).strip()}»?",
        ),
        (
            r"^([A-ZÁÉÍÓÚÑ][^,.;:]{3,65}?)\s+(?:se divide en|se clasifica en|comprende|incluye)\s+(.{15,})",
            lambda m: f"¿Cómo se clasifica o qué elementos incluye «{m.group(1).strip()}»?",
        ),
        (
            r"^([A-ZÁÉÍÓÚÑ][^,.;:]{3,65}?)\s+(?:se debe a|es causado por|es producido por|ocurre porque)\s+(.{15,})",
            lambda m: f"¿Cuál es la causa de «{m.group(1).strip()}»?",
        ),
    ]

    bad_starts = {"esto", "esta", "este", "sin", "ademas", "por", "en", "para", "segun", "tambien", "pero", "asimismo"}

    for pat, q_builder in patterns:
        match = re.search(pat, sentence.strip())
        if match:
            subject = match.group(1).strip()
            first_word = normalize_word(subject.split()[0])
            if first_word in bad_starts:
                continue
            return ExamQuestion(
                q_type="desarrollo",
                question=q_builder(match),
                answer=sentence,
                explanation="Respuesta modelo extraída directamente del documento.",
                source_section=section,
            )

    # Respaldo narrativo/conceptual para ensayos o literatura (si spaCy detecta personajes/entidades)
    if nlp is not None and not sentence.startswith(("—", "«")):
        doc = nlp(sentence)
        ents = [e for e in doc.ents if e.label_ in ("PER", "ORG", "LOC") and len(e.text.strip()) >= 4]
        if ents:
            ent = ents[0]
            if ent.label_ == "PER":
                q_text = f"Según el texto, ¿qué papel cumple o qué ocurre con «{ent.text.strip()}» en esta sección?"
            else:
                q_text = f"Explica la relevancia o lo mencionado sobre «{ent.text.strip()}» en el texto."
            return ExamQuestion(
                q_type="desarrollo",
                question=q_text,
                answer=sentence,
                explanation="Idea de referencia tomada del documento.",
                source_section=section,
            )

    return None


def _make_true_false(sentence: str, section: str, pool: List[str], keywords: List[str], nlp=None, make_false: bool = True) -> Optional[ExamQuestion]:
    if not make_false:
        return ExamQuestion(
            q_type="verdadero_falso",
            question=sentence,
            answer="Verdadero",
            options=["Verdadero", "Falso"],
            explanation="La afirmación coincide con lo expuesto en el documento.",
            source_section=section,
        )

    candidates = _extract_candidate_terms(sentence, keywords, nlp=nlp)
    if not candidates:
        return None
    target = candidates[0]
    distractors = _build_distractors(target, pool, n=1)
    if not distractors:
        return None

    fake_term = distractors[0]
    if target[0].isupper():
        fake_term = fake_term[0].upper() + fake_term[1:]

    mutated = re.sub(rf"\b{re.escape(target)}\b", fake_term, sentence, count=1)
    if mutated == sentence:
        return None

    return ExamQuestion(
        q_type="verdadero_falso",
        question=mutated,
        answer="Falso",
        options=["Verdadero", "Falso"],
        explanation=f"Falso. En el texto original el término correcto es «{target}» (no «{fake_term}»): {sentence}",
        source_section=section,
    )


def generate_exam_questions(
    summary: SummaryResult,
    num_questions: int = 10,
    selected_types: Optional[List[str]] = None,
    nlp=None,
    seed: int = 42,
) -> List[ExamQuestion]:
    if selected_types is None:
        selected_types = ["opcion_multiple", "completar", "desarrollo", "verdadero_falso"]

    rng = random.Random(seed)
    random.seed(seed)

    sentences = summary.selected_sentences
    if not sentences:
        return []

    keywords = [k for k, _ in summary.keywords]
    distractor_pool = list(keywords)
    for s in sentences:
        distractor_pool.extend(_extract_candidate_terms(s.text, keywords, nlp=nlp))

    questions: List[ExamQuestion] = []
    used_sentence_indices = set()

    # 1. Preguntas de desarrollo
    if "desarrollo" in selected_types:
        max_open = max(1, num_questions // max(len(selected_types), 1))
        for s in sentences:
            if len(questions) >= max_open:
                break
            q = _generate_open_question(s.text, s.section_title, nlp=nlp)
            if q:
                questions.append(q)
                used_sentence_indices.add(s.index)

    # 2. Repartir entre el resto de tipos
    remaining_types = [t for t in selected_types if t != "desarrollo"]
    remaining_sentences = [s for s in sentences if s.index not in used_sentence_indices]
    if len(remaining_sentences) < (num_questions - len(questions)):
        remaining_sentences.extend([s for s in sentences if s.index in used_sentence_indices])

    type_cursor = 0
    tf_toggle = True

    for s in remaining_sentences:
        if len(questions) >= num_questions or not remaining_types:
            break

        q_type = remaining_types[type_cursor % len(remaining_types)]
        type_cursor += 1
        candidates = _extract_candidate_terms(s.text, keywords, nlp=nlp)

        if q_type == "opcion_multiple" and candidates:
            target = candidates[0]
            distractors = _build_distractors(target, distractor_pool, n=3)
            if len(distractors) >= 2:
                blanked = re.sub(rf"\b{re.escape(target)}\b", "________", s.text, count=1)
                options = distractors + [target]
                rng.shuffle(options)
                questions.append(
                    ExamQuestion(
                        q_type="opcion_multiple",
                        question=f"Completa el enunciado con el concepto correcto:\n{blanked}",
                        answer=target,
                        options=options,
                        explanation=f"Oración original: {s.text}",
                        source_section=s.section_title,
                    )
                )
                continue

        if q_type == "completar" and candidates:
            target = candidates[0]
            blanked = re.sub(rf"\b{re.escape(target)}\b", "________", s.text, count=1)
            questions.append(
                ExamQuestion(
                    q_type="completar",
                    question=blanked,
                    answer=target,
                    explanation=f"Oración completa: {s.text}",
                    source_section=s.section_title,
                )
            )
            continue

        if q_type == "verdadero_falso":
            tf_q = _make_true_false(s.text, s.section_title, distractor_pool, keywords, nlp=nlp, make_false=tf_toggle)
            tf_toggle = not tf_toggle
            if tf_q:
                questions.append(tf_q)
                continue

    return questions[:num_questions]
