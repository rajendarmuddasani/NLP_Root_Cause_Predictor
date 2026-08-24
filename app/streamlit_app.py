"""
FA Failure Classifier — Streamlit v6
======================================
Binary EOS vs Non-EOS classification for specialist-lab routing decision.

Model : Word+Char TF-IDF + RF-SMOTE
        Macro F1 77.1% | EOS-Precision 69.5% | EOS-Recall 61.3% | Accuracy 83.3%
Classes: EOS → Route to Specialist Lab | Non-EOS → no specialist routing
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import os
from pathlib import Path
from datetime import datetime
import warnings
from scipy.sparse import hstack
import openai
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_DIR  = BASE_DIR / "models" / "v6"
RESULT_DIR = BASE_DIR / "results"
KW_DIR     = BASE_DIR / "keywords"

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ FA Failure Classifier",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS — compact zero-scroll layout ──────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 3rem !important; padding-bottom: 0.5rem !important; }
    .title-row { text-align: center; font-size: 1.6rem; font-weight: 700;
                 margin-bottom: 0.3rem; color: #1565C0; }
    .title-icon {
        display: inline-flex; align-items: center; justify-content: center;
        width: 36px; height: 36px; border-radius: 8px;
        background: linear-gradient(135deg, #1565C0, #42A5F5);
        color: #fff; font-size: 1.1rem; font-weight: 700;
        margin-right: 8px; vertical-align: middle;
        box-shadow: 0 2px 6px rgba(21,101,192,0.3);
    }
    .stats-bar { text-align: center; font-size: 0.78rem; color: #6c757d;
                 margin-bottom: 0.5rem; }
    .stats-bar b { color: #2c3e50; }
    div[data-testid="stButton"] > button { height: 68px !important; min-height: 68px !important; }
    /* Predict button — light blue */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #5DADE2 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-radius: 6px !important;
        transition: background-color 0.2s;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #2E86C1 !important;
    }
    .route-specialist {
        display: flex; align-items: center; justify-content: center;
        height: 68px; border-radius: 6px; font-weight: 700; font-size: 1.15rem;
        color: #fff;
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        text-align: center; width: 100%;
        box-shadow: 0 2px 8px rgba(231,76,60,0.3);
    }
    .route-standard {
        display: flex; align-items: center; justify-content: center;
        height: 68px; border-radius: 6px; font-weight: 700; font-size: 1.15rem;
        color: #fff;
        background: linear-gradient(135deg, #27ae60, #1e8449);
        text-align: center; width: 100%;
        box-shadow: 0 2px 8px rgba(39,174,96,0.3);
    }
    .route-idle {
        display: flex; align-items: center; justify-content: center;
        height: 68px; border-radius: 6px; font-weight: 600; font-size: 0.9rem;
        color: #888; background: #f0f0f0; border: 1px dashed #ccc;
        text-align: center; width: 100%;
    }
    .label-badge {
        display: inline-block; padding: 4px 14px; border-radius: 14px;
        font-weight: 700; font-size: 0.85rem; color: #fff;
        background: linear-gradient(135deg, #E65100, #FB8C00);
        box-shadow: 0 2px 6px rgba(230,81,0,0.25);
    }
    .tbl-header {
        font-size: 0.88rem; font-weight: 600; margin-bottom: 0.15rem;
        color: #1565C0; background: #E3F2FD; padding: 0.35rem 0.6rem;
        border-radius: 5px;
    }
    .tbl-explanation {
        font-size: 0.70rem; color: #6c757d; margin-top: 4px;
        padding: 0.35rem 0.45rem; background: #f8f9fa; border-radius: 4px;
        line-height: 1.45;
    }
    footer { visibility: hidden; }
    .stTable td, .stTable th {
        user-select: text !important; -webkit-user-select: text !important;
        cursor: text;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#   V6 MODEL METADATA (hardcoded — from v6 experiment results)
# ═══════════════════════════════════════════════════════════════════════════
V6_META = {
    "version": "v6",
    "model_name": "Word+Char TF-IDF + RF-SMOTE",
    "algorithm": "TF-IDF (word 1-2 + char_wb 3-5) → SMOTE → RandomForest",
    "macro_f1": 0.7705,
    "accuracy": 0.8325,
    "eos_precision": 0.6949,
    "eos_recall": 0.6127,
    "eos_f1": 0.6512,
    "delta_vs_v5a": "+0.0146",
    "tfidf_features": 6000,
    "train_rows": 11128,
    "test_rows": 2782,
    "classes": ["EOS", "Non-EOS"],
    "eos_class_index": 0,
}


# ═══════════════════════════════════════════════════════════════════════════
#   LOAD ARTEFACTS  (cached — loads once per session)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    """Load v6 model: separate word + char TF-IDF vectorizers, RF, LabelEncoder."""
    model   = joblib.load(MODEL_DIR / "best_model.joblib")
    tfidf_w = joblib.load(MODEL_DIR / "tfidf_word.joblib")
    tfidf_c = joblib.load(MODEL_DIR / "tfidf_char.joblib")
    le      = joblib.load(MODEL_DIR / "label_encoder.joblib")
    with open(MODEL_DIR / "metadata.json") as f:
        meta = json.load(f)
    return model, tfidf_w, tfidf_c, le, meta


@st.cache_data
def load_shap_summary():
    """Load pre-computed feature importances (top 100 per class)."""
    path = RESULT_DIR / "v5a_shap_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_keyword_list():
    """Load keyword reference table with EOS/Non-EOS rates."""
    path = KW_DIR / "keyword_list_v5a.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# ── Text cleaning — must exactly match NB preprocessing ──────────────────
def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9\s_\-/]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Keyword detection ─────────────────────────────────────────────────────
def detect_keywords(text: str, kw_df: pd.DataFrame, top_n: int = 10):
    """Find EOS-discriminant keywords in the input text (whole-word match)."""
    text_lower = text.lower()
    found = []
    for _, row in kw_df.iterrows():
        kw = str(row["keyword"]).lower()
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
            found.append({
                "Keyword":       kw,
                "EOS rate %":    round(float(row["eos_rate_pct"]), 1),
                "Non-EOS rate %": round(float(row["non_eos_rate_pct"]), 1),
            })
    found.sort(key=lambda r: abs(r["EOS rate %"] - r["Non-EOS rate %"]), reverse=True)
    return found[:top_n]


# ── Match input text to pre-computed top model features ──────────────────
def get_feature_matches(text: str, shap_df: pd.DataFrame, top_n: int = 8):
    """Return pre-computed top EOS features that appear in the input text."""
    text_lower = text.lower()
    eos_features = shap_df[shap_df['class'] == 'EOS'].head(150)
    matched = []
    for _, row in eos_features.iterrows():
        kw = str(row['keyword']).lower()
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
            matched.append({
                'Feature':    kw,
                'Importance': f"{float(row['mean_abs_shap']):.5f}",
            })
    return matched[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
#   MAIN APP
# ═══════════════════════════════════════════════════════════════════════════
def main():
    model, tfidf_w, tfidf_c, le, meta = load_model()
    shap_df = load_shap_summary()
    kw_df   = load_keyword_list()

    classes  = V6_META["classes"]
    eos_idx  = V6_META["eos_class_index"]
    thresh_b2  = float(meta.get("threshold_b2",  0.155))
    thresh_b05 = float(meta.get("threshold_b05", 0.700))

    # ── Session state ──────────────────────────────────────────────────────
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'threshold_slider' not in st.session_state:
        st.session_state['threshold_slider'] = 0.50

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Routing Threshold")

        st.markdown(
            "<div style='font-size:0.77rem;color:#6c757d;margin-bottom:6px;'>"
            "Quick presets:</div>",
            unsafe_allow_html=True,
        )
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            if st.button(
                "β=2\nRecall",
                use_container_width=True,
                help=(
                    f"📥  Set threshold to {thresh_b2:.3f}  (Recall-favoured β=2)\n\n"
                    "Lowers the decision boundary so the model flags almost any hint of EOS. "
                    "Maximises EOS cases caught (high recall) but sends more Non-EOS to Specialist Lab."
                ),
            ):
                st.session_state['threshold_slider'] = thresh_b2
                st.rerun()
        with pc2:
            if st.button(
                "Default\n0.500",
                use_container_width=True,
                help=(
                    "⚖️  Set threshold to 0.500  (Balanced)\n\n"
                    "Standard 50% probability cut-off. Equal trade-off."
                ),
            ):
                st.session_state['threshold_slider'] = 0.50
                st.rerun()
        with pc3:
            if st.button(
                "β=0.5\nPrecision",
                use_container_width=True,
                help=(
                    f"📤  Set threshold to {thresh_b05:.3f}  (Precision-favoured β=0.5)\n\n"
                    "Raises the decision boundary. Only high-confidence predictions Route to Specialist Lab."
                ),
            ):
                st.session_state['threshold_slider'] = thresh_b05
                st.rerun()

        active_threshold = st.slider(
            "EOS probability threshold",
            min_value=0.01,
            max_value=0.99,
            step=0.01,
            key='threshold_slider',
            help=(
                "Slide ◀ LEFT → more samples routed to Specialist Lab (higher recall).\n\n"
                "Slide ▶ RIGHT → fewer routed (higher precision)."
            ),
        )

        # Descriptive label
        if active_threshold <= 0.10:
            thr_label = "🔴 Very High Recall — Almost every case sent to Specialist Lab."
            thr_color = "#e74c3c"
        elif active_threshold <= 0.25:
            thr_label = "🟠 High Recall — Most borderline EOS caught."
            thr_color = "#e67e22"
        elif active_threshold <= 0.45:
            thr_label = "🟡 Recall-favoured — Good EOS sensitivity."
            thr_color = "#f39c12"
        elif active_threshold <= 0.55:
            thr_label = "⚖️ Balanced — Equal trade-off. Recommended default."
            thr_color = "#3498db"
        elif active_threshold <= 0.75:
            thr_label = "🟢 Precision-favoured — Fewer false alarms."
            thr_color = "#27ae60"
        else:
            thr_label = "🔵 High Precision — Only very confident EOS predictions."
            thr_color = "#2980b9"

        st.markdown(
            f'<div style="background:#f8f9fa;border-left:4px solid {thr_color};'
            f'padding:0.5rem 0.7rem;border-radius:4px;font-size:0.77rem;'
            f'margin-top:0.3rem;line-height:1.45;">'
            f'{thr_label}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:0.77rem;color:#6c757d;margin-top:6px;'>"
            f"Active cut-off: <b style='color:#1565C0;'>{active_threshold:.2f}</b></div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("**Model info (v6)**")
        st.markdown(f"- Model: `{V6_META['model_name']}`")
        st.markdown(f"- Macro F1: **{V6_META['macro_f1']:.4f}**")
        st.markdown(f"- Accuracy: **{V6_META['accuracy']:.4f}**")
        st.markdown(f"- EOS-Precision: **{V6_META['eos_precision']:.4f}**")
        st.markdown(f"- EOS-Recall: **{V6_META['eos_recall']:.4f}**")
        st.markdown(f"- vs v5a: **{V6_META['delta_vs_v5a']}** Macro-F1")
        st.markdown(f"- Train: {V6_META['train_rows']:,} rows")
        st.markdown(f"- Features: {V6_META['tfidf_features']:,} (word+char)")
        st.markdown("---")
        st.markdown(
            f"<div style='font-size:0.72rem;color:#999;'>"
            f"v6 = E1: word(1,2) + char_wb(3,5) TF-IDF → SMOTE → RF(200 trees)</div>",
            unsafe_allow_html=True,
        )

    # ── Title ─────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="title-row">'
        '<span class="title-icon">⚡</span>'
        'FA Failure Classifier'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Stats bar ─────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="stats-bar">'
        f'<b>v6</b> &nbsp;|&nbsp; '
        f'<b>Model:</b> {V6_META["model_name"]} &nbsp;|&nbsp; '
        f'<b>Accuracy:</b> {V6_META["accuracy"]*100:.1f}% &nbsp;|&nbsp; '
        f'<b>Macro F1:</b> {V6_META["macro_f1"]:.4f} &nbsp;|&nbsp; '
        f'<b>EOS F1:</b> {V6_META["eos_f1"]:.4f} &nbsp;|&nbsp; '
        f'<b>EOS Prec:</b> {V6_META["eos_precision"]:.4f} &nbsp;|&nbsp; '
        f'<b>EOS Rec:</b> {V6_META["eos_recall"]:.4f} &nbsp;|&nbsp; '
        f'<b>Threshold:</b> {active_threshold:.4f}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Input row: text box | predict button | route badge ────────────────
    col_input, col_btn, col_route = st.columns([5, 1, 2])

    with col_input:
        user_input = st.text_area(
            "Failure Description",
            height=68,
            placeholder="Enter failure description, e.g. 'short circuit at vddc pin, EOS damage observed' …",
            label_visibility="collapsed",
        )

    with col_btn:
        predict_clicked = st.button("🔮 Predict", use_container_width=True, type="primary")

    route_placeholder = col_route.empty()

    # ── Prediction logic ──────────────────────────────────────────────────
    if predict_clicked:
        if not user_input.strip():
            st.warning("⚠️ Please enter a failure description.")
            st.session_state.prediction = None
        else:
            cleaned = clean_text(user_input)
            X_vec   = hstack([tfidf_w.transform([cleaned]), tfidf_c.transform([cleaned])])
            proba    = model.predict_proba(X_vec)[0]
            eos_prob = float(proba[eos_idx])

            pred_cls = classes[eos_idx] if eos_prob >= active_threshold else classes[1 - eos_idx]

            kw_found      = detect_keywords(cleaned, kw_df) if kw_df is not None else []
            feat_matched  = get_feature_matches(cleaned, shap_df) if shap_df is not None else []

            st.session_state.prediction = {
                'input':        user_input,
                'cleaned':      cleaned,
                'pred_cls':     pred_cls,
                'eos_prob':     eos_prob,
                'proba':        proba.tolist(),
                'kw_found':     kw_found,
                'feat_matched': feat_matched,
                'threshold':    active_threshold,
                'timestamp':    datetime.now().strftime("%H:%M:%S"),
            }

    # ── Render results ────────────────────────────────────────────────────
    pred = st.session_state.prediction

    if pred is None:
        route_placeholder.markdown(
            '<div class="route-idle">EOS → Specialist Lab / Non-EOS</div>',
            unsafe_allow_html=True,
        )
        return

    pred_cls = pred['pred_cls']
    eos_prob = pred['eos_prob']
    proba    = pred['proba']
    thresh   = pred['threshold']
    is_eos   = (pred_cls == 'EOS')

    # ── Route badge ────────────────────────────────────────────────────────
    css_cls    = "route-specialist" if is_eos else "route-standard"
    route_text = "⚠️ EOS → Route to Specialist Lab" if is_eos else "✅ Non-EOS → Standard Lab"
    route_placeholder.markdown(
        f'<div class="{css_cls}">{route_text}</div>',
        unsafe_allow_html=True,
    )

    # ── Timestamp strip ────────────────────────────────────────────────────
    truncated = pred['cleaned'][:80] + ('…' if len(pred['cleaned']) > 80 else '')
    badge_cls = "label-badge"
    st.markdown(
        f'<div style="text-align:right;font-size:0.72rem;color:#999;'
        f'margin-top:-0.3rem;margin-bottom:0.2rem;">'
        f'<span class="{badge_cls}">{pred_cls}</span> &nbsp; '
        f'🕐 <b>{pred["timestamp"]}</b> &nbsp;|&nbsp; '
        f'Threshold: <b>{thresh:.4f}</b> &nbsp;|&nbsp; '
        f'Input: <i>{truncated}</i></div>',
        unsafe_allow_html=True,
    )

    # ── Three result columns ───────────────────────────────────────────────
    t1, t2, t3 = st.columns(3)

    # ── Col 1: EOS probability with visual bar ────────────────────────────
    with t1:
        st.markdown('<div class="tbl-header">📊 EOS Probability</div>',
                    unsafe_allow_html=True)
        eos_pct  = eos_prob * 100
        neos_pct = (1 - eos_prob) * 100
        eos_bar_color = "#e74c3c" if is_eos else "#c0392b"
        st.markdown(
            f'<div style="margin-bottom:8px;">'
            f'  <div style="font-size:0.8rem;margin-bottom:3px;">'
            f'    EOS: <b>{eos_pct:.1f}%</b> '
            f'    <span style="color:#999;font-size:0.73rem;">'
            f'    (threshold = {thresh*100:.1f}%)</span></div>'
            f'  <div style="background:#eee;border-radius:4px;height:20px;width:100%;">'
            f'    <div style="width:{min(eos_pct,100):.1f}%;background:{eos_bar_color};'
            f'    height:20px;border-radius:4px;"></div></div>'
            f'  <div style="font-size:0.8rem;margin-top:6px;margin-bottom:3px;">'
            f'    Non-EOS: <b>{neos_pct:.1f}%</b></div>'
            f'  <div style="background:#eee;border-radius:4px;height:20px;width:100%;">'
            f'    <div style="width:{min(neos_pct,100):.1f}%;background:#27ae60;'
            f'    height:20px;border-radius:4px;"></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.table(pd.DataFrame([
            {"Class": f"EOS {'◀' if is_eos else ''}",     "Probability": f"{eos_pct:.2f}%"},
            {"Class": f"Non-EOS {'◀' if not is_eos else ''}", "Probability": f"{neos_pct:.2f}%"},
        ]))
        # Explanation for probability table
        st.markdown(
            '<div class="tbl-explanation">'
            '<b>EOS Probability</b> — the model\'s confidence that this failure is EOS '
            '(Electrical Over-Stress). If ≥ threshold, the case is routed to Specialist Lab.<br>'
            'The probability comes from 200 RandomForest trees voting on 6,000 TF-IDF '
            'features (word + character n-grams). Higher = more EOS-like.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Col 2: EOS keywords detected in input ────────────────────────────
    with t2:
        st.markdown('<div class="tbl-header">🔑 EOS Keywords Detected</div>',
                    unsafe_allow_html=True)
        if pred['kw_found']:
            df_kw_display = pd.DataFrame(pred['kw_found']).rename(columns={
                'EOS rate %':     'EOS freq %',
                'Non-EOS rate %': 'Non-EOS freq %',
            })
            st.table(df_kw_display)
            st.markdown(
                '<div class="tbl-explanation">'
                '<b>EOS freq %</b> — how often this keyword appeared in <i>EOS-labelled</i> '
                'training samples (e.g. 25% means 1 in 4 EOS cases contained it).<br>'
                '<b>Non-EOS freq %</b> — same count for Non-EOS samples.<br>'
                'These are <b>independent dataset frequencies</b>, not probability '
                'contributions — they do <b>not</b> sum to the EOS probability above. '
                'The model score is computed by 200 decision trees voting across '
                '6,000 features simultaneously.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No domain keywords detected — description may be Non-EOS or too generic.")

    # ── Col 3: Top EOS model features matched in input ───────────────────
    with t3:
        st.markdown('<div class="tbl-header">🧪 EOS Model Features Matched</div>',
                    unsafe_allow_html=True)
        if pred['feat_matched']:
            st.table(pd.DataFrame(pred['feat_matched']))
            st.markdown(
                '<div class="tbl-explanation">'
                '<b>Feature</b> — a TF-IDF term (word or char n-gram) that the model '
                'learned is associated with EOS failures during training.<br>'
                '<b>Importance</b> — mean absolute SHAP value: how much this feature '
                'shifts the model\'s prediction toward EOS on average across test samples. '
                'Higher = stronger EOS signal. These are the top features from '
                'pre-computed SHAP analysis of the full test set.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No top EOS model features found in this input.")

    # ── GPT-Assisted Explanation ──────────────────────────────────────────
    st.markdown("---")
    gpt_col, _ = st.columns([3, 1])
    with gpt_col:
        st.markdown('<div class="tbl-header">🤖 GPT-Assisted Explanation</div>',
                    unsafe_allow_html=True)
        if st.button("Generate GPT Explanation", key="gpt_btn",
                     help="Uses GPT to explain why this failure description is or is not EOS."):
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                st.warning("OPENAI_API_KEY not set in environment. Add it to .env and restart.")
            else:
                kw_str  = ", ".join(k["Keyword"] for k in pred.get("kw_found", [])[:5]) or "none"
                cls     = pred["pred_cls"]
                prob    = pred["eos_prob"] * 100
                inp     = pred["input"][:300]
                prompt  = (
                    f"A semiconductor FA triage classifier predicted this failure description as '{cls}' "
                    f"with {prob:.1f}% EOS probability. "
                    f"EOS-related keywords found: {kw_str}.\n\n"
                    f"Description: \"{inp}\"\n\n"
                    f"In 3 concise sentences explain why this is {'likely EOS' if cls=='EOS' else 'likely not EOS'}, "
                    f"referencing the keywords and common EOS failure signatures."
                )
                try:
                    openai.api_key = api_key
                    resp = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an expert in semiconductor failure analysis."},
                            {"role": "user",   "content": prompt},
                        ],
                        max_tokens=200,
                        temperature=0.3,
                    )
                    explanation = resp["choices"][0]["message"]["content"].strip()
                    st.success(explanation)
                except Exception as exc:
                    st.error(f"GPT call failed: {exc}")
        else:
            st.caption("Click above to get a GPT explanation for the current prediction.")


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
