"""Motor de resumen proporcional por capítulos (TextRank + Heurísticas + MMR) optimizado para libros largos."""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.extractor import ExtractionResult
from core.nlp_utils import (
    SPANISH_STOPWORDS,
    normalize_word,
    split_sentences,
    tokenize_words,
)

DISCOURSE_BOOSTERS = [
    r"\bse define como\b",
    r"\bconsiste en\b",
    r"\ben conclusi[oó]n\b",
    r"\ben resumen\b",
    r"\bpor lo tanto\b",
    r"\bes fundamental\b",
    r"\bes importante\b",
    r"\bse caracteriza por\b",
    r"\bsu objetivo\b",
    r"\btiene como finalidad\b",
    r"\bse clasifica(n)? en\b",
    r"\bse divide(n)? en\b",
    r"\bpermite(n)?\b",
    r"\bventaja(s)?\b",
    r"\bdiferencia(s)?\b",
    r"\bprincipio(s)?\b",
    r"\bconcepto\b",
]


@dataclass
class ScoredSentence:
    index: int
    section_title: str
    text: str
    score: float


@dataclass
class SummaryResult:
    selected_sentences: List[ScoredSentence] = field(default_factory=list)
    by_section: Dict[str, List[str]] = field(default_factory=dict)
    keywords: List[Tuple[str, float]] = field(default_factory=list)
    glossary: List[Tuple[str, str]] = field(default_factory=list)
    total_sentences: int = 0
    total_words: int = 0
    total_sections: int = 0


def _pagerank(sim_matrix: np.ndarray, damping: float = 0.85, max_iter: int = 80, tol: float = 1e-5) -> np.ndarray:
    n = sim_matrix.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    mat = sim_matrix.copy()
    np.fill_diagonal(mat, 0.0)
    row_sums = mat.sum(axis=1, keepdims=True)
    transition = np.divide(mat, row_sums, out=np.full_like(mat, 1.0 / n), where=row_sums != 0)

    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = (1.0 - damping) / n + damping * transition.T.dot(scores)
        if np.linalg.norm(new_scores - scores, 1) < tol:
            scores = new_scores
            break
        scores = new_scores

    max_s = scores.max()
    if max_s > 0:
        scores = scores / max_s
    return scores


def _heuristic_score(sentence: str, pos_in_chunk: int, chunk_len: int, heading_tokens: set) -> float:
    score = 0.0
    if pos_in_chunk == 0:
        score += 0.28
    elif pos_in_chunk == chunk_len - 1 and chunk_len > 2:
        score += 0.14

    if heading_tokens:
        sent_tokens = {normalize_word(t) for t in tokenize_words(sentence)}
        overlap = len(sent_tokens & heading_tokens)
        if overlap > 0:
            score += min(0.35, 0.12 * overlap)

    lower = sentence.lower()
    marker_hits = sum(1 for pat in DISCOURSE_BOOSTERS if re.search(pat, lower))
    if marker_hits > 0:
        score += min(0.35, 0.18 * marker_hits)

    word_count = len(sentence.split())
    if word_count < 8:
        score -= 0.25
    elif word_count > 65:
        score -= 0.15

    # Penalizar ligeramente diálogos puros si hay narrativa/teoría disponible
    if sentence.startswith(("—", "«", "- ")) and len(sentence) < 70:
        score -= 0.12

    return max(0.0, score)


def _summarize_chunk(
    chunk_items: List[Tuple[int, str, str]],  # (global_idx, section_title, text)
    target_n: int,
    heading_tokens: set,
) -> List[ScoredSentence]:
    """Aplica TextRank + Heurística + MMR sobre un capítulo o bloque (<= 220 oraciones para ahorrar RAM)."""
    n = len(chunk_items)
    if n == 0:
        return []
    if n <= target_n:
        return [
            ScoredSentence(index=g_idx, section_title=sec_t, text=txt, score=1.0)
            for g_idx, sec_t, txt in chunk_items
        ]

    texts = [item[2] for item in chunk_items]
    vectorizer = TfidfVectorizer(
        stop_words=list(SPANISH_STOPWORDS),
        ngram_range=(1, 2),
        min_df=1,
        max_features=2000,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
        tfidf_matrix = vectorizer.fit_transform(texts)

    sim_matrix = cosine_similarity(tfidf_matrix)
    tr_scores = _pagerank(sim_matrix)

    combined = np.zeros(n)
    for i, (_, _, s_text) in enumerate(chunk_items):
        h_score = _heuristic_score(s_text, i, n, heading_tokens)
        combined[i] = 0.65 * tr_scores[i] + 0.35 * h_score

    # MMR para diversidad
    selected_local: List[int] = []
    candidates = list(range(n))
    lambda_param = 0.72

    while len(selected_local) < target_n and candidates:
        best_idx = None
        best_mmr = -1e9
        for c in candidates:
            rel = combined[c]
            red = 0.0 if not selected_local else max(sim_matrix[c, s] for s in selected_local)
            mmr = lambda_param * rel - (1.0 - lambda_param) * red
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = c
        selected_local.append(best_idx)
        candidates.remove(best_idx)

    selected_local.sort()
    return [
        ScoredSentence(
            index=chunk_items[i][0],
            section_title=chunk_items[i][1],
            text=chunk_items[i][2],
            score=float(combined[i]),
        )
        for i in selected_local
    ]


def _extract_global_keywords_and_glossary(
    all_sentences: List[Tuple[str, str]],
    top_k: int = 15,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, str]]]:
    """Extrae palabras clave globales y glosario usando muestra representativa para no saturar RAM."""
    if not all_sentences:
        return [], []

    # Si es un libro gigante (>1200 oraciones), muestrear uniformemente para TF-IDF global rápido
    sample = all_sentences
    if len(all_sentences) > 1200:
        step = max(1, len(all_sentences) // 1200)
        sample = all_sentences[::step]

    texts = [s[1] for s in sample]
    vectorizer = TfidfVectorizer(
        stop_words=list(SPANISH_STOPWORDS),
        ngram_range=(1, 2),
        min_df=1,
        max_features=3000,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return [], []

    feature_names = vectorizer.get_feature_names_out()
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    top_indices = mean_scores.argsort()[::-1]

    keywords: List[Tuple[str, float]] = []
    seen_norm = set()
    for idx in top_indices:
        term = feature_names[idx]
        norm = normalize_word(term)
        if len(norm) < 3 or norm.isdigit():
            continue
        if any(norm in s or s in norm for s in seen_norm):
            continue
        seen_norm.add(norm)
        keywords.append((term, float(mean_scores[idx])))
        if len(keywords) >= top_k:
            break

    glossary: List[Tuple[str, str]] = []
    used_sents = set()
    def_pattern = re.compile(r"\b(es|son|se define|consiste|significa|representa|refiere|permite)\b", re.IGNORECASE)

    for term, _ in keywords[:10]:
        best_sent = None
        best_rank = -1
        term_norm = normalize_word(term)
        for _, s_text in sample:
            if s_text in used_sents:
                continue
            if term_norm in normalize_word(s_text):
                rank = 1
                if def_pattern.search(s_text):
                    rank += 2
                if 10 <= len(s_text.split()) <= 45:
                    rank += 1
                if rank > best_rank:
                    best_rank = rank
                    best_sent = s_text
        if best_sent:
            used_sents.add(best_sent)
            glossary.append((term.capitalize(), best_sent))

    return keywords, glossary


def summarize_document(
    extraction: ExtractionResult,
    ratio: float = 0.25,
    max_sentences: Optional[int] = None,
    nlp=None,
) -> SummaryResult:
    """
    Genera un resumen proporcional al tamaño del documento, procesando sección por sección
    (y dividiendo secciones gigantes en bloques de <= 220 oraciones para estabilidad en Streamlit Cloud).
    """
    section_buckets: List[Tuple[str, List[Tuple[int, str, str]]]] = []
    all_flat: List[Tuple[str, str]] = []
    global_idx = 0

    for sec in extraction.sections:
        sec_text = "\n\n".join(sec.paragraphs)
        sents = split_sentences(sec_text, nlp=nlp)
        if not sents:
            continue

        # Si una sola sección tiene demasiadas oraciones (ej. un libro sin capítulos detectados), dividir en partes
        max_chunk = 220
        if len(sents) > max_chunk:
            num_parts = (len(sents) + max_chunk - 1) // max_chunk
            for p_idx in range(num_parts):
                sub = sents[p_idx * max_chunk:(p_idx + 1) * max_chunk]
                part_title = f"{sec.title} — Parte {p_idx + 1}"
                bucket = []
                for s in sub:
                    bucket.append((global_idx, part_title, s))
                    all_flat.append((part_title, s))
                    global_idx += 1
                section_buckets.append((part_title, bucket))
        else:
            bucket = []
            for s in sents:
                bucket.append((global_idx, sec.title, s))
                all_flat.append((sec.title, s))
                global_idx += 1
            section_buckets.append((sec.title, bucket))

    total_sents = len(all_flat)
    total_words = sum(len(s[1].split()) for s in all_flat)

    if total_sents == 0:
        return SummaryResult(total_sentences=0, total_words=0, total_sections=0)

    # Calcular objetivo proporcional automático si max_sentences es None
    if max_sentences is None:
        # Escala suave: documentos cortos ~25-30%, libros largos se mantienen manejables (hasta ~80 oraciones)
        if total_sents <= 60:
            global_target = max(4, int(round(total_sents * ratio)))
        elif total_sents <= 300:
            global_target = max(12, min(45, int(round(total_sents * ratio))))
        else:
            global_target = max(25, min(80, int(round(total_sents * min(ratio, 0.12)))))
    else:
        global_target = max(3, min(max_sentences, int(round(total_sents * ratio))))

    global_target = min(global_target, total_sents)

    # Repartir presupuesto de oraciones proporcionalmente a cada sección/capítulo
    selected_all: List[ScoredSentence] = []
    for sec_title, bucket in section_buckets:
        sec_heading_tokens = {normalize_word(t) for t in tokenize_words(sec_title)}
        proportion = len(bucket) / total_sents
        sec_target = max(1, int(round(global_target * proportion)))
        sec_target = min(sec_target, len(bucket))
        selected_all.extend(_summarize_chunk(bucket, sec_target, sec_heading_tokens))

    # Si por redondeo se pasó demasiado del límite manual, recortar las de menor puntaje manteniendo orden
    if max_sentences is not None and len(selected_all) > max_sentences:
        selected_all.sort(key=lambda x: x.score, reverse=True)
        selected_all = selected_all[:max_sentences]

    selected_all.sort(key=lambda x: x.index)

    by_section: Dict[str, List[str]] = {}
    for item in selected_all:
        by_section.setdefault(item.section_title, []).append(item.text)

    keywords, glossary = _extract_global_keywords_and_glossary(all_flat)

    return SummaryResult(
        selected_sentences=selected_all,
        by_section=by_section,
        keywords=keywords,
        glossary=glossary,
        total_sentences=total_sents,
        total_words=total_words,
        total_sections=len(section_buckets),
    )
