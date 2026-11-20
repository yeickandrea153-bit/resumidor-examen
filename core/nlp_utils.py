"""Utilidades de Procesamiento de Lenguaje Natural (NLP) en español v2.0 sin APIs externas."""
import re
import unicodedata
from typing import List, Set

SPANISH_STOPWORDS: Set[str] = {
    "a", "acá", "ahí", "al", "algo", "algún", "alguna", "algunas", "alguno", "algunos",
    "allá", "allí", "ambos", "ante", "antes", "aquel", "aquella", "aquellas", "aquello",
    "aquellos", "aquí", "arriba", "así", "atrás", "aun", "aunque", "bajo", "bastante",
    "bien", "cabe", "cada", "casi", "cierto", "cierta", "ciertas", "ciertos", "como",
    "cómo", "con", "conmigo", "conseguimos", "conseguir", "consigo", "consigue",
    "consiguen", "consigues", "contigo", "contra", "cual", "cuál", "cuales", "cuáles",
    "cualquier", "cualquiera", "cuando", "cuándo", "cuanto", "cuánto", "cuanta",
    "cuánta", "cuantas", "cuántas", "cuantos", "cuántos", "de", "debajo", "debe",
    "deben", "deber", "debido", "decir", "del", "delante", "demás", "demasiado",
    "dentro", "desde", "después", "detrás", "día", "días", "dice", "dicen", "dicho",
    "diferente", "diferentes", "donde", "dónde", "dos", "durante", "e", "ejemplo",
    "el", "él", "ella", "ellas", "ello", "ellos", "embargo", "emplean", "emplear",
    "empleas", "en", "encima", "entonces", "entre", "era", "erais", "éramos", "eran",
    "eras", "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta", "está", "estaba",
    "estabais", "estábamos", "estaban", "estabas", "estad", "estada", "estadas",
    "estado", "estados", "estamos", "están", "estando", "estar", "estará", "estarán",
    "estarás", "estaré", "estaréis", "estaremos", "estaría", "estarían", "estas",
    "estás", "este", "esto", "estos", "estoy", "estuve", "estuviera", "estuvieron",
    "estuvo", "etc", "fin", "fue", "fueron", "fui", "fuimos", "gran", "grande", "grandes",
    "ha", "había", "habían", "haber", "habrá", "hace", "hacen", "hacer", "hacia",
    "haciendo", "han", "hasta", "hay", "haya", "he", "hecho", "hemos", "hicieron",
    "hizo", "hoy", "igual", "incluso", "ir", "la", "las", "le", "les", "lo", "los",
    "luego", "manera", "más", "mayor", "me", "mediante", "mejor", "menos", "mi", "mí",
    "mía", "mías", "mientras", "mío", "míos", "mis", "misma", "mismas", "mismo",
    "mismos", "modo", "mucha", "muchas", "mucho", "muchos", "muy", "nada", "nadie",
    "ni", "ningún", "ninguna", "ningunas", "ninguno", "ningunos", "no", "nos",
    "nosotras", "nosotros", "nuestra", "nuestras", "nuestro", "nuestros", "nueva",
    "nuevas", "nuevo", "nuevos", "nunca", "o", "os", "otra", "otras", "otro", "otros",
    "para", "parece", "parte", "pero", "poca", "pocas", "poco", "pocos", "podemos",
    "poder", "podría", "podrían", "poner", "por", "porque", "porqué", "posible",
    "primer", "primera", "primero", "primeros", "propia", "propias", "propio",
    "propios", "puede", "pueden", "puedo", "pues", "que", "qué", "quien", "quién",
    "quienes", "quiénes", "quiere", "quizá", "quizás", "realizar", "realizó",
    "respecto", "sabe", "saben", "saber", "se", "sé", "sea", "sean", "según", "ser",
    "será", "serán", "sería", "serían", "si", "sí", "sido", "siempre", "siendo",
    "sigue", "siguiente", "sin", "sino", "sobre", "sola", "solamente", "solas", "solo",
    "sólo", "solos", "somos", "son", "soy", "su", "sus", "suya", "suyas", "suyo",
    "suyos", "tal", "tales", "también", "tampoco", "tan", "tanta", "tantas", "tanto",
    "tantos", "te", "tendrá", "tendrán", "tenemos", "tener", "tenga", "tengan", "tengo",
    "tenía", "tenían", "tercer", "ti", "tiempo", "tiene", "tienen", "toda", "todas",
    "todavía", "todo", "todos", "total", "tras", "través", "tres", "tu", "tú", "tus",
    "tuyo", "último", "un", "una", "unas", "uno", "unos", "usa", "usan", "usar", "uso",
    "usted", "ustedes", "va", "vais", "valor", "vamos", "van", "varias", "varios",
    "vaya", "veces", "ver", "vez", "vosotras", "vosotros", "vuestra", "vuestras",
    "vuestro", "vuestros", "y", "ya", "yo"
}

ABBREVIATIONS = (
    r"Sr|Sra|Srta|Dr|Dra|Ing|Lic|Mg|Prof|Fig|Cap|Art|Pág|Pag|Vol|Núm|Num|No|Ej|Etc|vs|p|pp|ed|cfr"
)

# Palabras con las que una oración completa NO debería terminar
DANGLING_ENDINGS = {
    "y", "e", "o", "u", "que", "de", "del", "la", "el", "los", "las", "un", "una",
    "en", "con", "por", "para", "sin", "sobre", "entre", "hacia", "desde", "pero",
    "sino", "aunque", "porque", "como", "cuando", "donde", "si", "ni", "al", "su", "sus",
}

# Títulos de secciones que deben excluirse del resumen y del examen
EXCLUDED_SECTION_PATTERNS = [
    r"^(referencias|bibliograf[ií]a|referencias bibliogr[aá]ficas|fuentes consultadas)$",
    r"^([ií]ndice|tabla de contenido(s)?|contenido|sumario)$",
    r"^(anexos?|ap[eé]ndices?|agradecimientos?|dedicatoria)$",
]

# Patrones para detectar títulos de capítulos o unidades en texto/PDF
CHAPTER_HEADING_PATTERNS = [
    r"^(cap[ií]tulo|unidad|tema|m[oó]dulo|secci[oó]n|parte|lecci[oó]n)\s+[0-9ivxlcdm]+[\s:.\-–—]*.*$",
    r"^([IVXLCDM]{1,5})\.\s+[A-ZÁÉÍÓÚÑ].{2,80}$",
    r"^(\d{1,2}(\.\d{1,2}){0,2})\.?\s+[A-ZÁÉÍÓÚÑ].{3,85}$",
]


def load_spacy_model():
    """Carga el modelo español de spaCy ya instalado (vía requirements.txt) sin intentar pip install en vivo."""
    try:
        import spacy
        return spacy.load("es_core_news_sm")
    except Exception:
        return None


def normalize_word(word: str) -> str:
    """Minúsculas y sin tildes para comparaciones robustas."""
    w = word.lower().strip()
    w = "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn")
    return w


def is_excluded_section(title: str) -> bool:
    """Devuelve True si el título corresponde a Índice, Referencias, Bibliografía o Anexos."""
    clean_t = re.sub(r"^[\d.\s\-–—]+", "", title.strip()).lower()
    return any(re.match(pat, clean_t) for pat in EXCLUDED_SECTION_PATTERNS)


def is_chapter_heading(line: str) -> bool:
    """Detecta si una línea parece un título de capítulo, unidad, subtítulo numerado o sección estándar."""
    s = line.strip()
    if not s or len(s) < 4 or len(s) > 95:
        return False
    if is_excluded_section(s):
        return True
    # No debe terminar en punto final típico de oración larga ni en coma
    if s.endswith((",", ";", ":")):
        return False
    for pat in CHAPTER_HEADING_PATTERNS:
        if re.match(pat, s, re.IGNORECASE):
            return True
    # Título corto totalmente en mayúsculas (evitando siglas sueltas o ruido)
    words = s.split()
    if s.isupper() and 2 <= len(words) <= 10 and sum(c.isalpha() for c in s) >= 6:
        return True
    return False


def is_toc_or_bibliography_line(line: str) -> bool:
    """Filtra líneas de tabla de contenido (.... 12) o citas bibliográficas puras."""
    s = line.strip()
    if not s:
        return True
    # Líneas de índice con puntos suspensivos y número de página al final
    if re.search(r"\.{4,}\s*\d+\s*$", s):
        return True
    # Solo número de página o "Página X de Y"
    if re.match(r"^(p[aá]g(ina)?\.?\s*)?\d+(\s*(de|/)\s*\d+)?$", s, re.IGNORECASE):
        return True
    if re.match(r"^[-–—\s]*\d+[-–—\s]*$", s):
        return True
    # Citas APA/IEEE típicas (Apellido, A. (2020)...) o URLs/DOIs sueltos
    if re.match(r"^[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+,\s+[A-Z]\.\s*(\([12]\d{3}\))", s):
        return True
    if re.search(r"(https?://\S+|doi\.org/\S+|ISSN|ISBN)", s, re.IGNORECASE) and len(s) < 220:
        return True
    return False


def clean_text(text: str) -> str:
    """Limpia ruido típico de PDFs/Word/OCR manteniendo saltos de párrafo."""
    if not text:
        return ""
    # Unir palabras cortadas con guion al final de línea
    text = re.sub(r"([a-zA-ZáéíóúÁÉÍÓÚñÑ])-\s*\n\s*([a-zA-ZáéíóúÁÉÍÓÚñÑ])", r"\1\2", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def refold_paragraphs(text: str) -> str:
    """Reconstruye párrafos uniendo renglones rotos de PDFs cuando una línea no cierra oración."""
    text = clean_text(text)
    if not text:
        return ""

    raw_lines = [l.strip() for l in text.split("\n")]
    paragraphs: List[str] = []
    buffer: List[str] = []

    def flush():
        if buffer:
            joined = " ".join(buffer)
            joined = re.sub(r"\s+", " ", joined).strip()
            if joined:
                paragraphs.append(joined)
            buffer.clear()

    for line in raw_lines:
        if not line:
            flush()
            continue

        if is_toc_or_bibliography_line(line):
            continue

        # Si parece encabezado o viñeta nueva, cerrar párrafo anterior
        if is_chapter_heading(line) or re.match(r"^([•●▪◦\-–—*]|\d+[.)])\s+", line):
            flush()
            buffer.append(line)
            if is_chapter_heading(line):
                flush()
            continue

        if not buffer:
            buffer.append(line)
        else:
            prev = buffer[-1]
            # Si la línea anterior NO termina en puntuación de cierre, es continuación del mismo párrafo
            if not re.search(r'[.!?»"\']\s*$', prev):
                buffer.append(line)
            # Si la línea actual empieza con minúscula o coma/guion, también es continuación
            elif line[0].islower() or line[0] in ",;:)—–":
                buffer.append(line)
            else:
                # Si la línea anterior era muy corta (posible salto de párrafo real), separar
                if len(prev) < 60:
                    flush()
                    buffer.append(line)
                else:
                    buffer.append(line)

    flush()
    return "\n\n".join(paragraphs)


def is_valid_sentence(sentence: str) -> bool:
    """Filtra fragmentos rotos, diálogos cortados o líneas basura."""
    s = sentence.strip()
    if len(s) < 25:
        return False
    words = s.split()
    if len(words) < 5:
        return False

    # Descartar si empieza en minúscula (fragmento cortado de oración previa)
    first_alpha = next((c for c in s if c.isalpha()), None)
    if first_alpha is None or first_alpha.islower():
        return False

    # Descartar si termina en coma o conjunción/preposición colgada ("...el mundo que", "...uno, y")
    if s.endswith((",", ";", ":", "—", "–", "-")):
        return False
    last_word = re.sub(r"[^a-zA-ZáéíóúÁÉÍÓÚñÑ]", "", words[-1]).lower()
    if last_word in DANGLING_ENDINGS:
        return False

    # Debe contener una proporción razonable de letras (evitar tablas crudas o ruido OCR)
    alpha_ratio = sum(c.isalpha() for c in s) / max(len(s), 1)
    if alpha_ratio < 0.55:
        return False

    return True


def is_clean_ocr_text(text: str) -> bool:
    """Filtro de calidad para descartar basura visual de OCR (marcas de agua o ruido)."""
    if not text or len(text.strip()) < 15:
        return False
    s = text.strip()
    if looks_like_code(s):
        return True

    # Proporción de caracteres alfanuméricos + espacios
    valid_chars = sum(c.isalnum() or c.isspace() or c in ".,;:¿?¡!()-%/áéíóúÁÉÍÓÚñÑ" for c in s)
    if (valid_chars / max(len(s), 1)) < 0.72:
        return False

    tokens = re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ]+", s)
    if len(tokens) < 3:
        return False
    # Si más de la mitad son letras sueltas de 1-2 caracteres, suele ser ruido de gráfico/diagrama
    short_ratio = sum(1 for t in tokens if len(t) <= 2) / len(tokens)
    if short_ratio > 0.55:
        return False

    return True


def tokenize_words(text: str, remove_stopwords: bool = True) -> List[str]:
    """Extrae tokens alfabéticos limpios."""
    tokens = re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]{3,}", text.lower())
    if remove_stopwords:
        return [t for t in tokens if t not in SPANISH_STOPWORDS]
    return tokens


def split_sentences(text: str, nlp=None) -> List[str]:
    """Divide texto ya reconstruido en oraciones completas y válidas."""
    rebuilt = refold_paragraphs(text)
    if not rebuilt:
        return []

    raw_sentences: List[str] = []

    if nlp is not None:
        # Procesar en bloques seguros para no exceder max_length de spaCy en libros grandes
        for paragraph in rebuilt.split("\n\n"):
            p = paragraph.strip()
            if not p:
                continue
            # Cortar párrafos monstruosos si existieran
            for chunk_start in range(0, len(p), 50000):
                chunk = p[chunk_start:chunk_start + 50000]
                doc = nlp(chunk)
                for sent in doc.sents:
                    raw_sentences.append(sent.text.strip())
    else:
        abbr_pattern = re.compile(rf"\b({ABBREVIATIONS})\.", re.IGNORECASE)
        for paragraph in rebuilt.split("\n\n"):
            p = paragraph.strip()
            if not p:
                continue
            protected = abbr_pattern.sub(r"\1<DOT>", p)
            protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)
            parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡«\"0-9])", protected)
            for part in parts:
                raw_sentences.append(part.replace("<DOT>", ".").strip())

    # Filtrar oraciones rotas o incompletas
    return [s for s in raw_sentences if is_valid_sentence(s)]


def looks_like_code(text: str) -> bool:
    """Heurística para distinguir bloques de código (de OCR o texto) de prosa teórica."""
    if not text or len(text.strip()) < 15:
        return False
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False

    code_patterns = [
        r"\b(def|class|import|from|return|public|private|static|void|int|float|string|var|let|const|function|SELECT|INSERT|UPDATE|DELETE|CREATE TABLE|WHERE|for|while|if|else|elif|switch|case|try|catch|print|console\.log|System\.out|echo|printf|ufw|sudo|apt|git|docker)\b",
        r"[{};\[\]]",
        r"==|!=|<=|>=|=>|->|\+=|-=",
        r"//.*$|/\*|\*/|#include|#!/",
        r"\w+\s*\([^)]*\)\s*[:{]?",
    ]

    code_line_hits = 0
    for line in lines:
        hits = sum(1 for pat in code_patterns if re.search(pat, line))
        symbol_ratio = len(re.findall(r"[{}()\[\];=<>+\-*/_\\|&]", line)) / max(len(line), 1)
        if hits >= 2 or symbol_ratio > 0.14:
            code_line_hits += 1

    return (code_line_hits / len(lines)) >= 0.35
