"""ACOS Course Evaluation System · Streamlit App
Supports English and Bahasa Melayu input.
BM input → translated to English → ACOS inference → 4-column bilingual result table.
"""

import os, io, re, time, random
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="ACOS · Course Evaluation",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Previous deployed model path: ../The Best Model/
HF_MODEL_ID = "lijinzheyy/acos-flan-t5-course-evaluation"
LOCAL_MODEL_DIR = os.path.join(APP_DIR, "models", "final_flan_t5_acos_model")

MODEL_DIR = os.getenv("ACOS_MODEL_DIR")

if not MODEL_DIR:
    if os.path.isdir(LOCAL_MODEL_DIR):
        MODEL_DIR = LOCAL_MODEL_DIR
    else:
        MODEL_DIR = HF_MODEL_ID

SAMPLES_EN = [
    ("Screenshot workload", "The lecturer explained the concepts clearly, but the workload was too heavy."),
    ("Instructor clarity",  "The instructor explains concepts very clearly and responds to questions quickly. Best course I've taken so far."),
    ("Mixed feedback",      "The instructor explains concepts clearly, but the assignments are far too vague and the grading is inconsistent."),
    ("Pacing issue",        "Excellent course content! The topics are relevant, but the pacing is way too fast for beginners."),
    ("Poor support",        "Forum support is very slow and the lecture notes are completely disorganised. Very disappointed."),
    ("Positive overall",    "The hands-on projects are practical and well-designed. Highly recommended for anyone in data science."),
    ("Grading unclear",     "Great instructor who responds quickly to emails. The grading criteria is unclear and confusing though."),
    ("Content heavy",       "The course content is too heavy and the weekly readings take forever. The quizzes are fair though."),
    ("Bad teaching",        "Bad teaching, random, disorganized. Hardly any explanation of concepts. You'll learn almost nothing."),
    ("Easy and useful",     "Useful and easy to follow. A bit dry in terms of delivery, but the material itself is solid."),
    ("Highly negative",     "He was very rude and did not respect his students. I would not take any more of his courses."),
]

SAMPLES_MS = [
    ("Pensyarah bagus",         "Pensyarah menjelaskan konsep dengan sangat jelas dan mudah difahami. Kursus terbaik yang pernah saya ikuti sejauh ini."),
    ("Tugasan tidak jelas",     "Pensyarah baik dan peramah, tetapi tugasan terlalu kabur dan markah tidak konsisten. Agak mengecewakan."),
    ("Kandungan terlalu berat", "Kandungan kursus terlalu berat dan bacaan mingguan mengambil masa yang sangat lama. Namun begitu, kuiz adalah adil."),
    ("Sokongan forum lemah",    "Sokongan forum sangat lambat dan nota kuliah tidak tersusun langsung. Saya sangat kecewa dengan kursus ini."),
    ("Projek praktikal",        "Projek hands-on sangat praktikal dan direka dengan baik. Saya sangat mengesyorkan kursus ini untuk sesiapa dalam bidang sains data."),
    ("Pengajaran buruk",        "Pengajaran sangat buruk, tidak tersusun dan tiada penjelasan konsep yang mencukupi. Anda hampir tidak akan belajar apa-apa."),
    ("Kadar pembelajaran laju", "Kandungan kursus sangat relevan tetapi kadar pembelajaran terlalu laju untuk pelajar baru dalam bidang ini."),
    ("Maklum balas campuran",   "Pensyarah sangat baik dan membalas emel dengan cepat, tetapi kriteria penilaian tidak jelas dan mengelirukan."),
]

PLACEHOLDER_EN = "— Select English sample —"
PLACEHOLDER_MS = "— Pilih sampel Bahasa Melayu —"

MARIAN_MS_EN = "Helsinki-NLP/opus-mt-mul-en"

# Common Malay abbreviations / hyphenated words to normalise before translation
MS_NORMALISE = [
    (r'\be-mel\b', 'emel'),
    (r'\be-learning\b', 'elearning'),
    (r'\be-book\b', 'ebook'),
    (r'\bhand-on\b', 'hands on'),
]

# ── Static EN→BM dictionary for ACOS table cells ─────────────────────────────
BM_DICT = {
    "positive": "positif", "negative": "negatif", "neutral": "neutral",
    "implicit": "tersirat",
    "course": "kursus", "course general": "kursus am",
    "course quality": "kualiti kursus", "course content": "kandungan kursus",
    "instructor": "pensyarah", "lecturer": "pensyarah", "teacher": "guru",
    "assignment": "tugasan", "assignments": "tugasan",
    "grading": "penilaian", "grade": "gred", "grades": "gred",
    "grading criteria": "kriteria penilaian",
    "content": "kandungan", "material": "bahan", "materials": "bahan",
    "pacing": "kadar pembelajaran", "pace": "kadar",
    "support": "sokongan", "forum": "forum", "forum support": "sokongan forum",
    "quiz": "kuiz", "quizzes": "kuiz",
    "project": "projek", "projects": "projek",
    "lecture notes": "nota kuliah", "notes": "nota",
    "reading": "bacaan", "readings": "bacaan",
    "teaching": "pengajaran",
    "clear": "jelas", "very clear": "sangat jelas", "clearly": "dengan jelas",
    "good": "baik", "very good": "sangat baik",
    "excellent": "cemerlang", "great": "hebat",
    "practical": "praktikal", "well-designed": "direka dengan baik",
    "useful": "berguna", "easy": "mudah", "easy to follow": "mudah diikuti",
    "solid": "mantap", "relevant": "relevan", "very relevant": "sangat relevan",
    "fair": "adil", "recommended": "disyorkan",
    "quick": "cepat", "quickly": "dengan cepat",
    "friendly": "mesra", "helpful": "membantu",
    "organised": "tersusun", "organized": "tersusun",
    "well organised": "tersusun dengan baik",
    "slow": "lambat", "very slow": "sangat lambat",
    "fast": "laju", "too fast": "terlalu laju",
    "heavy": "berat", "too heavy": "terlalu berat",
    "vague": "kabur", "too vague": "terlalu kabur",
    "unclear": "tidak jelas", "confusing": "mengelirukan",
    "inconsistent": "tidak konsisten",
    "disorganised": "tidak tersusun", "disorganized": "tidak tersusun",
    "disappointed": "kecewa", "very disappointed": "sangat kecewa",
    "disappointing": "mengecewakan",
    "bad": "buruk", "poor": "lemah",
    "rude": "kasar", "random": "tidak tentu hala",
    "short": "pendek", "too short": "terlalu pendek",
    "long": "lama", "forever": "terlalu lama",
}

def to_bm(text: str) -> str:
    key = text.lower().strip()
    return BM_DICT.get(key, text)

def sanitize_filename(name: str) -> str:
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^\w]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "batch"

def normalise_ms(text: str) -> str:
    """Normalise common Malay hyphenated/abbreviated forms before translation."""
    for pattern, replacement in MS_NORMALISE:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def offline_ms_en_translation(text: str) -> str:
    """Small offline fallback for built-in Malay demo reviews when MarianMT is unavailable."""
    cleaned = re.sub(r"\s+", " ", normalise_ms(text)).strip()
    key = cleaned.lower()
    fallback_map = {
        "pensyarah menjelaskan konsep dengan sangat jelas dan mudah difahami. kursus terbaik yang pernah saya ikuti sejauh ini.":
            "The lecturer explained the concepts very clearly and was easy to understand. This is the best course I have taken so far.",
        "pensyarah baik dan peramah, tetapi tugasan terlalu kabur dan markah tidak konsisten. agak mengecewakan.":
            "The lecturer was kind and friendly, but the assignments were too unclear and the marks were inconsistent. It was quite disappointing.",
        "kandungan kursus terlalu berat dan bacaan mingguan mengambil masa yang sangat lama. namun begitu, kuiz adalah adil.":
            "The course content was too heavy and the weekly readings took a very long time. However, the quizzes were fair.",
        "sokongan forum sangat lambat dan nota kuliah tidak tersusun langsung. saya sangat kecewa dengan kursus ini.":
            "The forum support was very slow and the lecture notes were not organized at all. I was very disappointed with this course.",
        "projek hands-on sangat praktikal dan direka dengan baik. saya sangat mengesyorkan kursus ini untuk sesiapa dalam bidang sains data.":
            "The hands-on project was very practical and well designed. I strongly recommend this course for anyone in the data science field.",
        "pengajaran sangat buruk, tidak tersusun dan tiada penjelasan konsep yang mencukupi. anda hampir tidak akan belajar apa-apa.":
            "The teaching was very poor, not organized, and had insufficient concept explanation. You will hardly learn anything.",
        "kandungan kursus sangat relevan tetapi kadar pembelajaran terlalu laju untuk pelajar baru dalam bidang ini.":
            "The course content was very relevant but the learning pace was too fast for new students in this field.",
        "pensyarah sangat baik dan membalas emel dengan cepat, tetapi kriteria penilaian tidak jelas dan mengelirukan.":
            "The lecturer was very good and replied to emails quickly, but the assessment criteria were unclear and confusing.",
    }
    return fallback_map.get(key, text)

def looks_like_malay(text):
    """
    Simple Malay keyword fallback for short course-review inputs.
    Used when automatic language detection fails.
    """
    if not isinstance(text, str):
        return False

    text_lower = text.lower()

    malay_keywords = [
        "pensyarah", "kursus", "kuliah", "tutorial", "tugasan",
        "markah", "peperiksaan", "ujian", "kelas", "pelajar",
        "baik", "peramah", "kabur", "susah", "mudah", "jelas",
        "tidak", "terlalu", "agak", "mengecewakan", "membantu",
        "tetapi", "dan", "dengan", "untuk", "dalam", "kerana"
    ]

    hit_count = sum(1 for word in malay_keywords if word in text_lower)

    return hit_count >= 2

def is_valid_translation(result: str, original: str) -> bool:
    """Return False if the translation looks like garbage (mostly dots/symbols)."""
    if not result or not result.strip():
        return False
    alpha_chars = sum(1 for c in result if c.isalpha())
    alpha_ratio = alpha_chars / max(len(result), 1)
    if alpha_ratio < 0.4:
        return False
    # If result is nearly identical to input (not translated), treat as failure
    if result.strip().lower() == original.strip().lower():
        return False
    return True

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; font-size: 16px; }
p, li, span, div, label { font-size: 15px !important; }
textarea { font-size: 15px !important; }
.stButton > button { font-size: 15px !important; padding: 0.45rem 1rem !important; }
h1 { font-size: 1.9rem !important; } h2 { font-size: 1.4rem !important; }
h3 { font-size: 1.15rem !important; } h4 { font-size: 1rem !important; }

[data-testid="stSidebar"] { background: #0f172a !important; border-right: 1px solid #1e293b; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; font-size: 14px !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #94a3b8 !important; }
[data-testid="stSidebar"] hr { border-color: #334155 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #1e293b !important; color: #e2e8f0 !important;
    border: 1px solid #334155 !important; border-radius: 6px; width: 100%;
    font-size: 14px !important; padding: 0.4rem 0.7rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover { background: #334155 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1e293b !important; color: #e2e8f0 !important; border: 1px solid #334155 !important;
}
.sb-info-card {
    background: #1e293b; border: 1px solid #334155; border-radius: 8px;
    padding: 0.8rem 1rem; font-size: 13px !important; color: #cbd5e1 !important; line-height: 2;
}
.sb-info-card b { color: #f1f5f9 !important; }

.main-header { padding: 1.4rem 0 0.9rem; border-bottom: 2px solid #e2e8f0; margin-bottom: 1.4rem; }
.main-header h1 { font-size: 1.9rem !important; font-weight: 700; color: #1e293b; margin: 0; }
.main-header p  { font-size: 1.05rem !important; color: #64748b; margin: 0.25rem 0 0; }

.how-grid { display:flex; gap:1rem; margin:1.2rem 0; flex-wrap:wrap; }
.how-card { flex:1; min-width:180px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:1.1rem 1.2rem; }
.how-card .step { font-size:1.6rem; margin-bottom:0.4rem; }
.how-card h4 { margin:0 0 0.35rem; font-size:1rem !important; font-weight:600; color:#1e293b; }
.how-card p  { margin:0; font-size:0.88rem !important; color:#64748b; line-height:1.55; }

.lang-badge { display:inline-block; padding:4px 12px; border-radius:5px; font-size:0.85rem !important; font-weight:600; margin-bottom:0.7rem; }
.lang-en { background:#dbeafe; color:#1d4ed8; }
.lang-ms { background:#fef9c3; color:#854d0e; }

.status-bar {
    display:flex; align-items:center; gap:8px; background:#f0fdf4; border:1px solid #bbf7d0;
    border-radius:7px; padding:0.6rem 1.1rem; font-size:0.92rem !important; color:#166534; margin-bottom:1.2rem;
}
.trans-note { font-size:0.85rem !important; color:#64748b; background:#fffbeb; border:1px solid #fde68a; border-radius:6px; padding:8px 14px; margin-bottom:1rem; }

.acos-table { width:100%; border-collapse:separate; border-spacing:0; border-radius:10px; overflow:hidden; border:1px solid #e2e8f0; font-size:0.9rem !important; }
.acos-table thead tr { background:#f1f5f9; }
.acos-table thead th { padding:11px 16px; text-align:left; font-weight:600; color:#475569; font-size:0.8rem !important; letter-spacing:.05em; text-transform:uppercase; border-bottom:1px solid #e2e8f0; }
.acos-table tbody tr { background:#fff; transition:background 0.1s; }
.acos-table tbody tr:hover { background:#f8fafc; }
.acos-table tbody tr:not(:last-child) td { border-bottom:1px solid #f1f5f9; }
.acos-table tbody td { padding:10px 16px; color:#334155; vertical-align:middle; }
.acos-table tbody td:first-child { font-weight:500; color:#1e293b; }
.cell-en { display:block; font-weight:500; color:#1e293b; }
.cell-bm { display:block; font-size:0.82rem !important; color:#92400e; margin-top:3px; font-style:italic; }
.implicit-tag { color:#94a3b8; font-style:italic; }
.sent-chip { display:inline-block; padding:4px 13px; border-radius:20px; font-size:0.82rem !important; font-weight:600; color:#fff; }
.sent-chip small { display:block; font-size:0.72rem !important; opacity:0.9; font-style:italic; }
.sent-positive { background:#16a34a; }
.sent-negative { background:#dc2626; }
.sent-neutral  { background:#94a3b8; }

.stat-row { display:flex; gap:1rem; margin-bottom:1.2rem; flex-wrap:wrap; }
.stat-box { flex:1; min-width:110px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:0.85rem 1rem; text-align:center; }
.stat-box .val { font-size:1.6rem !important; font-weight:700; color:#1e293b; }
.stat-box .lbl { font-size:0.8rem !important; color:#94a3b8; margin-top:3px; }

[data-testid="stTabs"] [role="tab"] { font-weight:500; font-size:0.97rem !important; color:#64748b; border-bottom:2px solid transparent; padding-bottom:6px; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:#2563eb; border-bottom-color:#2563eb; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "model": None, "tokenizer": None, "model_loaded": False,
    "ms_en_tok": None, "ms_en_mdl": None,
    "translators_loaded": False, "trans_error": "",
    "review_text": "", "batch_results": [],
    "upload_filename": "batch", "ta_key": 0, "last_sample_label": "",
    "last_loaded_en": PLACEHOLDER_EN,
    "last_loaded_ms": PLACEHOLDER_MS,
    "en_sel_ver": 0,
    "ms_sel_ver": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_dir):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    tok = T5Tokenizer.from_pretrained(model_dir)
    mdl = T5ForConditionalGeneration.from_pretrained(model_dir)
    mdl.eval()
    return tok, mdl

@st.cache_resource(show_spinner=False)
def load_ms_en_model():
    from transformers import MarianMTModel, MarianTokenizer
    tok = MarianTokenizer.from_pretrained(MARIAN_MS_EN)
    mdl = MarianMTModel.from_pretrained(MARIAN_MS_EN)
    mdl.eval()
    return tok, mdl

def ensure_translators():
    if st.session_state.translators_loaded:
        return
    try:
        tok, mdl = load_ms_en_model()
        st.session_state.ms_en_tok         = tok
        st.session_state.ms_en_mdl         = mdl
        st.session_state.translators_loaded = True
        st.session_state.trans_error       = ""
    except Exception as e:
        st.session_state.trans_error        = f"{type(e).__name__}: {e}"
        st.session_state.translators_loaded = True

def translate_ms_en(text: str) -> str:
    """Malay → English. Normalises text first, validates output, falls back to original."""
    tok = st.session_state.ms_en_tok
    mdl = st.session_state.ms_en_mdl
    if tok is None or mdl is None or not text.strip():
        return offline_ms_en_translation(text)
    try:
        import torch
        cleaned = normalise_ms(text)

        # Try with >>ms<< language tag first
        for tagged in [f">>ms<< {cleaned}", cleaned]:
            inputs = tok([tagged], return_tensors="pt",
                         padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                out = mdl.generate(**inputs, num_beams=4, max_length=512)
            result = tok.decode(out[0], skip_special_tokens=True)
            if is_valid_translation(result, text):
                return result

        return offline_ms_en_translation(text)
    except Exception:
        return offline_ms_en_translation(text)

def detect_language(text: str) -> str:
    if looks_like_malay(text):
        return "ms"
    try:
        from langdetect import detect
        return "ms" if detect(text) in ("ms", "id") else "en"
    except Exception:
        return "en"

# ── Inference ─────────────────────────────────────────────────────────────────
def clean_field(s: str) -> str:
    return s.strip().lstrip(";").strip().lstrip("(").strip().rstrip(")").strip()

def parse_output(raw: str):
    quads = []
    for part in raw.split(")"):
        part = clean_field(part)
        if not part:
            continue
        fields = [clean_field(f) for f in part.split("|")]
        while len(fields) < 4:
            fields.append("implicit")
        quads.append(tuple(fields[:4]))
    return quads

def run_inference(text, tokenizer, model):
    import torch
    prompt  = f"Extract ACOS quadruples from this course review: {text}"
    inputs  = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, num_beams=4, early_stopping=True)
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return raw, parse_output(raw)

def render_table_en(quads):
    rows = ""
    for asp, cat, op, sent in quads:
        s  = sent.lower()
        sc = "positive" if "pos" in s else ("negative" if "neg" in s else "neutral")
        asp_h = f'<span class="implicit-tag">{asp}</span>' if asp == "implicit" else asp
        op_h  = f'<span class="implicit-tag">{op}</span>'  if op  == "implicit" else op
        rows += (f"<tr><td>{asp_h}</td><td>{cat}</td><td>{op_h}</td>"
                 f"<td><span class='sent-chip sent-{sc}'>{sent}</span></td></tr>")
    return (f'<table class="acos-table"><thead><tr>'
            f'<th>Aspect</th><th>Category</th><th>Opinion</th><th>Sentiment</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>')

def render_table_bilingual(quads):
    sent_bm = {"positive": "positif", "negative": "negatif", "neutral": "neutral"}
    rows = ""
    for asp, cat, op, sent in quads:
        s  = sent.lower()
        sc = "positive" if "pos" in s else ("negative" if "neg" in s else "neutral")

        if asp == "implicit":
            asp_cell = '<span class="implicit-tag">implicit</span><span class="cell-bm">tersirat</span>'
        else:
            asp_bm = to_bm(asp)
            asp_cell = (f'<span class="cell-en">{asp}</span>'
                        + (f'<span class="cell-bm">{asp_bm}</span>' if asp_bm != asp else ''))

        cat_bm = to_bm(cat)
        cat_cell = (f'<span class="cell-en">{cat}</span>'
                    + (f'<span class="cell-bm">{cat_bm}</span>' if cat_bm != cat else ''))

        if op == "implicit":
            op_cell = '<span class="implicit-tag">implicit</span><span class="cell-bm">tersirat</span>'
        else:
            op_bm = to_bm(op)
            op_cell = (f'<span class="cell-en">{op}</span>'
                       + (f'<span class="cell-bm">{op_bm}</span>' if op_bm != op else ''))

        sent_cell = (f"<span class='sent-chip sent-{sc}'>{sent}"
                     f"<small>{sent_bm.get(sc, sent)}</small></span>")

        rows += (f"<tr><td>{asp_cell}</td><td>{cat_cell}</td>"
                 f"<td>{op_cell}</td><td>{sent_cell}</td></tr>")

    return (f'<table class="acos-table"><thead><tr>'
            f'<th>Aspect / Aspek</th><th>Category / Kategori</th>'
            f'<th>Opinion / Pendapat</th><th>Sentiment / Sentimen</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>')

def set_review(text: str, label: str = ""):
    st.session_state.review_text       = text
    st.session_state.last_sample_label = label
    st.session_state.ta_key           += 1

# ── Auto-load model ───────────────────────────────────────────────────────────
if not st.session_state.model_loaded:
    try:
        tok, mdl = load_model(MODEL_DIR)
        st.session_state.tokenizer    = tok
        st.session_state.model        = mdl
        st.session_state.model_loaded = True
    except Exception:
        pass

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 ACOS System")
    st.markdown("Aspect-Category-Opinion-Sentiment extraction for course reviews.")
    st.markdown("---")

    st.markdown("**Model Status**")
    if st.session_state.model_loaded:
        st.success("✓ ACOS model loaded")
    else:
        st.warning("Not loaded — check model files")

    st.markdown("**Language / Bahasa**")
    st.markdown("🇬🇧 English · 🇲🇾 Bahasa Melayu")

    st.markdown("---")
    st.markdown("**Demo Controls**")
    if st.button("▶ Run All Samples"):
        st.session_state["run_all"] = True
    if st.button("🎲 Random Sample"):
        pick_label, pick_text = random.choice(SAMPLES_EN + SAMPLES_MS)
        set_review(pick_text, pick_label)
        st.session_state.last_loaded_en = PLACEHOLDER_EN
        st.session_state.last_loaded_ms = PLACEHOLDER_MS
        st.session_state.en_sel_ver += 1
        st.session_state.ms_sel_ver += 1
        st.rerun()
    if st.button("✕ Clear"):
        set_review("", "")
        st.session_state.last_loaded_en = PLACEHOLDER_EN
        st.session_state.last_loaded_ms = PLACEHOLDER_MS
        st.session_state.en_sel_ver += 1
        st.session_state.ms_sel_ver += 1
        st.session_state.batch_results = []
        st.rerun()

    if st.session_state.get("trans_error"):
        st.markdown("---")
        st.markdown("**⚠️ Translation Error**")
        with st.expander("Show error details"):
            st.code(st.session_state.trans_error, language="text")
        if st.button("🔄 Retry"):
            load_ms_en_model.clear()
            st.session_state.translators_loaded = False
            st.session_state.trans_error = ""
            st.rerun()

    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown(f"""
<div class="sb-info-card">
  ACOS: <b>flan-t5-base</b><br>
  Epochs: <b>8</b> · Final F1: <b>0.3137</b><br>
  Translation: <b>Helsinki-NLP MarianMT</b><br>
  Dir: <b>{os.path.basename(MODEL_DIR)}/</b>
</div>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🎓 ACOS Course Evaluation System</h1>
  <p>Aspect-Category-Opinion-Sentiment structured extraction &nbsp;·&nbsp; 🇬🇧 English &nbsp;·&nbsp; 🇲🇾 Bahasa Melayu</p>
</div>
""", unsafe_allow_html=True)

tab_single, tab_batch, tab_perf, tab_about = st.tabs(
    ["Single Review", "Batch Analysis", "Performance", "About"]
)

# ════════════════════════════════════════════════════════════
# TAB 1 · Single Review
# ════════════════════════════════════════════════════════════
with tab_single:
    if st.session_state.model_loaded:
        st.markdown('<div class="status-bar">● Model ready — auto-detects English / Bahasa Melayu · bilingual 4-column results for Melayu input</div>',
                    unsafe_allow_html=True)
    else:
        st.error("Model not loaded. Ensure model files are in models/final_flan_t5_acos_model/.")

    st.markdown("**How it works**")
    st.markdown("""
<div class="how-grid">
  <div class="how-card">
    <div class="step">✍️</div>
    <h4>1 · Enter a Review</h4>
    <p>Paste any course review in <b>English</b> or <b>Bahasa Melayu</b>, or pick a built-in sample.</p>
  </div>
  <div class="how-card">
    <div class="step">🌐</div>
    <h4>2 · Language Detection</h4>
    <p>Bahasa Melayu input is auto-detected and translated to English via Helsinki-NLP MarianMT.</p>
  </div>
  <div class="how-card">
    <div class="step">⚙️</div>
    <h4>3 · ACOS Inference</h4>
    <p>FLAN-T5-base fine-tuned on OATS-ABSA extracts Aspect · Category · Opinion · Sentiment.</p>
  </div>
  <div class="how-card">
    <div class="step">📊</div>
    <h4>4 · Bilingual Results</h4>
    <p>4-column ACOS table — each cell shows the English term and its Bahasa Melayu equivalent below.</p>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    col_input, col_samples = st.columns([3, 1])

    with col_samples:
        st.markdown("**Sample Reviews**")

        st.caption("🇬🇧 English")
        labels_en = [PLACEHOLDER_EN] + [s[0] for s in SAMPLES_EN]
        chosen_en = st.selectbox(
            "EN", labels_en,
            key=f"sel_en_{st.session_state.en_sel_ver}",
            label_visibility="collapsed",
        )
        if chosen_en != PLACEHOLDER_EN and chosen_en != st.session_state.last_loaded_en:
            idx = [s[0] for s in SAMPLES_EN].index(chosen_en)
            set_review(SAMPLES_EN[idx][1], chosen_en)
            st.session_state.last_loaded_en = chosen_en
            st.session_state.last_loaded_ms = PLACEHOLDER_MS
            st.session_state.ms_sel_ver += 1
            st.rerun()

        st.caption("🇲🇾 Bahasa Melayu")
        labels_ms = [PLACEHOLDER_MS] + [s[0] for s in SAMPLES_MS]
        chosen_ms = st.selectbox(
            "MS", labels_ms,
            key=f"sel_ms_{st.session_state.ms_sel_ver}",
            label_visibility="collapsed",
        )
        if chosen_ms != PLACEHOLDER_MS and chosen_ms != st.session_state.last_loaded_ms:
            idx = [s[0] for s in SAMPLES_MS].index(chosen_ms)
            set_review(SAMPLES_MS[idx][1], chosen_ms)
            st.session_state.last_loaded_ms = chosen_ms
            st.session_state.last_loaded_en = PLACEHOLDER_EN
            st.session_state.en_sel_ver += 1
            st.rerun()

        if st.session_state.last_sample_label:
            st.caption(f"Loaded: *{st.session_state.last_sample_label}*")

    with col_input:
        review_input = st.text_area(
            "Course Review (English or Bahasa Melayu)",
            value=st.session_state.review_text,
            height=180,
            placeholder="Paste a course review here…\nTampal ulasan kursus di sini…",
            key=f"ta_{st.session_state.ta_key}",
        )
        analyse_clicked = st.button("Analyse →", type="primary")

    if analyse_clicked and review_input.strip():
        if not st.session_state.model_loaded:
            st.error("Model is not loaded.")
        else:
            with st.spinner("Detecting language and running inference…"):
                t0   = time.time()
                lang = detect_language(review_input.strip())

                if lang == "ms":
                    ensure_translators()
                    offline_translation = offline_ms_en_translation(review_input.strip())
                    if st.session_state.trans_error and offline_translation == review_input.strip():
                        st.warning("⚠️ Translation failed — see **Translation Error** in sidebar.")
                    translated_en  = translate_ms_en(review_input.strip())
                    if translated_en != review_input.strip():
                        st.session_state.trans_error = ""
                    inference_text = translated_en
                else:
                    translated_en  = None
                    inference_text = review_input.strip()

                raw, quads = run_inference(
                    inference_text,
                    st.session_state.tokenizer,
                    st.session_state.model,
                )
                elapsed = time.time() - t0

            if lang == "ms":
                st.markdown('<span class="lang-badge lang-ms">🇲🇾 Bahasa Melayu — bilingual ACOS table</span>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="trans-note">📝 <b>Terjemahan (EN):</b> {translated_en}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<span class="lang-badge lang-en">🇬🇧 English</span>',
                            unsafe_allow_html=True)

            st.markdown(f"**{len(quads)} quadruple(s) extracted** — _{elapsed:.2f}s_")

            if quads:
                if lang == "ms":
                    st.markdown(render_table_bilingual(quads), unsafe_allow_html=True)
                else:
                    st.markdown(render_table_en(quads), unsafe_allow_html=True)
            else:
                st.info("No valid quadruples parsed from model output.")

            with st.expander("Raw model output"):
                st.code(raw)

    elif analyse_clicked:
        st.warning("Please enter a review.")

# ════════════════════════════════════════════════════════════
# TAB 2 · Batch Analysis
# ════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("#### Batch Analysis")
    st.markdown("Upload a CSV/Excel file. Mixed English/Bahasa Melayu rows are handled automatically.")

    run_all_flag = st.session_state.pop("run_all", False)
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="batch_upload")

    if uploaded:
        st.session_state.upload_filename = sanitize_filename(os.path.splitext(uploaded.name)[0])

    col_run, col_demo, _ = st.columns([1, 1, 4])
    run_btn  = col_run.button("▶ Run Batch", type="primary")
    demo_btn = col_demo.button("Run Demo Samples")

    pending_texts = None

    if demo_btn or run_all_flag:
        pending_texts = [s[1] for s in SAMPLES_EN + SAMPLES_MS]
        st.session_state.upload_filename = "demo"

    elif run_btn and uploaded:
        try:
            if uploaded.name.endswith((".xlsx", ".xls")):
                df_up = pd.read_excel(uploaded)
            else:
                raw_bytes = uploaded.read()
                df_up = None
                for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"]:
                    try:
                        df_up = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
                        break
                    except Exception:
                        continue
                if df_up is None:
                    raise ValueError("Unable to decode file.")
            text_cands = [c for c in df_up.columns
                          if any(k in c.lower() for k in ["text","review","sentence","comment","ulasan"])]
            default_col = text_cands[0] if text_cands else df_up.select_dtypes(include="object").columns[0]
            col_sel, _ = st.columns([2, 3])
            with col_sel:
                text_col = st.selectbox("Text column", df_up.columns.tolist(),
                                        index=df_up.columns.tolist().index(default_col))
            pending_texts = df_up[text_col].dropna().astype(str).tolist()
            st.caption(f"{len(pending_texts)} rows loaded from **{text_col}**")
        except Exception as e:
            st.error(f"Could not read file: {e}")

    elif uploaded:
        try:
            if uploaded.name.endswith((".xlsx", ".xls")):
                df_up = pd.read_excel(uploaded)
            else:
                raw_bytes = uploaded.read()
                df_up = None
                for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"]:
                    try:
                        df_up = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
                        break
                    except Exception:
                        continue
            if df_up is not None:
                text_cands = [c for c in df_up.columns
                              if any(k in c.lower() for k in ["text","review","sentence","comment","ulasan"])]
                default_col = text_cands[0] if text_cands else df_up.select_dtypes(include="object").columns[0]
                col_sel, _ = st.columns([2, 3])
                with col_sel:
                    st.selectbox("Text column", df_up.columns.tolist(),
                                 index=df_up.columns.tolist().index(default_col))
                st.caption(f"{len(df_up)} rows — click **Run Batch** to analyse")
        except Exception:
            pass

    if pending_texts:
        if not st.session_state.model_loaded:
            st.error("Model is not loaded.")
        else:
            ensure_translators()
            results = []
            prog = st.progress(0, text="Running batch…")
            for i, txt in enumerate(pending_texts):
                lang   = detect_language(txt)
                txt_en = translate_ms_en(txt) if lang == "ms" else txt
                raw, quads = run_inference(txt_en, st.session_state.tokenizer, st.session_state.model)
                sent_bm_map = {"positive":"positif","negative":"negatif","neutral":"neutral"}
                for q in quads:
                    s  = q[3].lower()
                    sc = "positive" if "pos" in s else ("negative" if "neg" in s else "neutral")
                    if lang == "ms":
                        asp_bm = "tersirat" if q[0]=="implicit" else to_bm(q[0])
                        op_bm  = "tersirat" if q[2]=="implicit" else to_bm(q[2])
                        cat_bm = to_bm(q[1])
                    else:
                        asp_bm = op_bm = cat_bm = ""
                    results.append({
                        "Review":        txt[:80] + ("…" if len(txt) > 80 else ""),
                        "Language":      "BM→EN" if lang == "ms" else "EN",
                        "Aspect (EN)":   q[0],  "Aspek (BM)":    asp_bm,
                        "Category (EN)": q[1],  "Kategori (BM)": cat_bm,
                        "Opinion (EN)":  q[2],  "Pendapat (BM)": op_bm,
                        "Sentiment":     q[3],  "Sentimen":      sent_bm_map.get(sc,"") if lang=="ms" else "",
                    })
                prog.progress((i+1)/len(pending_texts), text=f"{i+1}/{len(pending_texts)}")
            prog.empty()
            st.session_state.batch_results = results

    if st.session_state.batch_results:
        df_res = pd.DataFrame(st.session_state.batch_results)

        def colour_sentiment(val):
            v = str(val).lower()
            if "pos" in v: return "color:#16a34a; font-weight:600"
            if "neg" in v: return "color:#dc2626; font-weight:600"
            return "color:#64748b"

        def style_sentiment_column(df):
            """
            Style the Sentiment column in a pandas-version-compatible way.
            New pandas versions use Styler.map; older pandas versions use Styler.applymap.
            """
            styler = df.style

            if "Sentiment" not in df.columns:
                return styler

            if hasattr(styler, "map"):
                return styler.map(colour_sentiment, subset=["Sentiment"])

            return styler.applymap(colour_sentiment, subset=["Sentiment"])

        st.dataframe(style_sentiment_column(df_res),
                     use_container_width=True, height=420, hide_index=True)
        fname_stem    = st.session_state.get("upload_filename", "batch")
        download_name = f"{fname_stem}_ACOS_Analysis.csv"
        st.download_button("⬇ Download CSV",
                           data=df_res.to_csv(index=False).encode("utf-8-sig"),
                           file_name=download_name, mime="text/csv")
        st.caption(f"Saved as: **{download_name}**")

# ════════════════════════════════════════════════════════════
# TAB 3 · Performance
# ════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown("#### Final Model Performance")

    st.markdown("""
<div class="stat-row">
  <div class="stat-box"><div class="val">0.3137</div><div class="lbl">Final F1</div></div>
  <div class="stat-box"><div class="val">0.3341</div><div class="lbl">Precision</div></div>
  <div class="stat-box"><div class="val">0.2957</div><div class="lbl">Recall</div></div>
  <div class="stat-box"><div class="val">8</div><div class="lbl">Epochs Trained</div></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("#### Model Comparison")
    df_comp = pd.DataFrame([
        {
            "Model": "Random Baseline",
            "Method Type": "Rule-free baseline",
            "Precision": "—",
            "Recall": "—",
            "F1 Score": "0.0000",
            "Notes": "Randomly predicts ACOS tuples",
        },
        {
            "Model": "TF-IDF + kNN Baseline",
            "Method Type": "Traditional ML baseline",
            "Precision": "—",
            "Recall": "—",
            "F1 Score": "0.0579",
            "Notes": "Retrieves similar training examples using TF-IDF + kNN",
        },
        {
            "Model": "Old deployed model",
            "Method Type": "FLAN-T5 previous deployed model",
            "Precision": "0.2504",
            "Recall": "0.2147",
            "F1 Score": "0.2312",
            "Notes": "Previous app model with post-processing",
        },
        {
            "Model": "Final retrained model",
            "Method Type": "FLAN-T5-base final retrained model",
            "Precision": "0.3341",
            "Recall": "0.2957",
            "F1 Score": "0.3137",
            "Notes": "Final model with post-processing",
        },
    ])
    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    st.caption("Random and TF-IDF + kNN are baseline systems. The final retrained FLAN-T5-base model achieves the best tuple-level Exact Match F1 with post-processing on the all-domain OATS-ABSA test set.")

# ════════════════════════════════════════════════════════════
# TAB 4 · About
# ════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
#### About This System

**ACOS Course Evaluation System** is a natural-language processing prototype that performs structured sentiment analysis on student course reviews. It supports both **English** and **Bahasa Melayu** input.

---

**Language Pipeline**

| Step | Detail |
|------|--------|
| Detection | `langdetect` identifies Bahasa Melayu (`ms`) vs English (`en`) |
| BM → EN | `Helsinki-NLP/opus-mt-mul-en` (MarianMT multilingual → English) |
| ACOS inference | `FLAN-T5-base` fine-tuned model extracts quadruples in English |
| EN → BM (table) | Static bilingual dictionary — clean, symbol-free display |
| Output | 4-column ACOS table — each cell shows EN term + BM equivalent below |

> **First-time use:** The MS→EN model (~300 MB) downloads automatically on first Bahasa Melayu input. Cached locally thereafter.

---

**Underlying ACOS Model**

- **Base:** FLAN-T5-base · **Input:** `Extract ACOS quadruples from this course review: {text}`
- **Output:** `(aspect | category | opinion | sentiment)`
- **Training data:** OATS-ABSA processed train split (16,415 examples, 23,204 ACOS tuples)
- **Setup:** 8 epochs · AdamW · batch 8 · final model saved separately
- **Evaluation:** all-domain OATS-ABSA test set · F1 = 0.3137 · P = 0.3341 · R = 0.2957
- **EduRABSA:** inspected for future education-domain expansion, but not used in current training

---

**Demo Samples**

10 English + 8 Bahasa Melayu course reviews covering: instructor quality, pacing, grading, content load, forum support, and overall experience. Not from the training or test sets.

---

**Limitations**

- ACOS model trained on English only — Bahasa Melayu accuracy depends on translation quality.
- Bilingual table uses a curated dictionary; novel opinion words show English only.
- Exact Match F1 is strict; partial matches score zero.
- Research prototype — not for production deployment without further validation.

---

**Tech Stack**

`Python 3.10` · `Streamlit` · `HuggingFace Transformers` · `PyTorch` · `Pandas` · `langdetect`
· `Helsinki-NLP/opus-mt-mul-en` (MarianMT)
""")
