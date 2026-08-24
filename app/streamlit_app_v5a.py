"""
FA Failure Classifier — Streamlit v5a
======================================
Binary EOS vs Non-EOS classification for specialist-lab routing decision.

Model : RF-SMOTE — Macro F1 75.6%, EOS-Precision 64.3%, EOS-Recall 62.7%
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
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_DIR  = BASE_DIR / "models" / "v5a"
RESULT_DIR = BASE_DIR / "results"
KW_DIR     = BASE_DIR / "keywords"

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🤖 FA Failure Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS — compact zero-scroll layout ──────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 3rem !important; padding-bottom: 0.5rem !important; }
    .title-row { text-align: center; font-size: 1.6rem; font-weight: 700;
                 margin-bottom: 0.3rem; color: #1565C0; }
    .stats-bar { text-align: center; font-size: 0.78rem; color: #6c757d;
                 margin-bottom: 0.5rem; }
    .stats-bar b { color: #2c3e50; }
    div[data-testid="stButton"] > button { height: 68px !important; min-height: 68px !important; }
    /* Predict button — light blue */
    div[data-testid="column"].predict-btn-col div[data-testid="stButton"] > button {
        background-color: #5DADE2 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-radius: 6px !important;
        transition: background-color 0.2s;
    }
    div[data-testid="column"].predict-btn-col div[data-testid="stButton"] > button:hover {
        background-color: #2E86C1 !important;
    }
    .route-specialist {
        display: flex; align-items: center; justify-content: center;
        height: 68px; border-radius: 6px; font-weight: 700; font-size: 1.15rem;
        color: #fff; background: #e74c3c; text-align: center; width: 100%;
    }
    .route-standard {
        display: flex; align-items: center; justify-content: center;
        height: 68px; border-radius: 6px; font-weight: 700; font-size: 1.15rem;
        color: #fff; background: #27ae60; text-align: center; width: 100%;
    }
    .route-idle {
        display: flex; align-items: center; justify-content: center;
        height: 68px; border-radius: 6px; font-weight: 600; font-size: 0.9rem;
        color: #888; background: #f0f0f0; border: 1px dashed #ccc;
        text-align: center; width: 100%;
    }
    .tbl-header {
        font-size: 0.88rem; font-weight: 600; margin-bottom: 0.15rem;
        color: #1565C0; background: #E3F2FD; padding: 0.35rem 0.6rem;
        border-radius: 5px;
    }
    footer { visibility: hidden; }
    .stTable td, .stTable th {
        user-select: text !important; -webkit-user-select: text !important;
        cursor: text;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#   LOAD ARTEFACTS  (cached — loads once per session)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    """Load best classical model + TF-IDF + LabelEncoder + metadata."""
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    le    = joblib.load(MODEL_DIR / "label_encoder.joblib")
    with open(MODEL_DIR / "metadata.json") as f:
        meta = json.load(f)
    # Load scaler only if needed (NB models)
    scaler = None
    if meta.get("uses_nb_scaler", False):
        scaler_path = MODEL_DIR / "maxabs_scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
    return model, tfidf, le, meta, scaler


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
    # Sort by EOS discriminative strength (difference)
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
    model, tfidf, le, meta, scaler = load_model()
    shap_df = load_shap_summary()
    kw_df   = load_keyword_list()

    classes  = meta.get("classes", list(le.classes_))   # ['EOS', 'Non-EOS']
    eos_idx  = meta.get("eos_class_index", 0)
    thresh_b2  = float(meta.get("threshold_b2",  0.036))
    thresh_b05 = float(meta.get("threshold_b05", 0.525))

    # ── Session state ──────────────────────────────────────────────────────
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'threshold_slider' not in st.session_state:
        st.session_state['threshold_slider'] = 0.50

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Routing Threshold")

        # ── Quick-preset buttons ───────────────────────────────────────────
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
                    f"📥  Set threshold to {thresh_b2:.3f}  (Recall-favoured  β=2)\n\n"
                    "Lowers the decision boundary so the model flags almost any hint of EOS. "
                    "This maximises the number of true EOS cases caught (high recall) but also "
                    "sends more Non-EOS samples to Specialist Lab as false alarms.\n\n"
                    "Best when: missing a real EOS case is costly or unacceptable."
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
                    "The standard 50 % probability cut-off. The model treats catching an EOS "
                    "case and avoiding a false alarm with equal importance.\n\n"
                    "Best when: no strong preference for recall or precision."
                ),
            ):
                st.session_state['threshold_slider'] = 0.50
                st.rerun()
        with pc3:
            if st.button(
                "β=0.5\nPrecision",
                use_container_width=True,
                help=(
                    f"📤  Set threshold to {thresh_b05:.3f}  (Precision-favoured  β=0.5)\n\n"
                    "Raises the decision boundary so only high-confidence predictions are "
                    "routed to Specialist Lab. Reduces Non-EOS false alarms, but some genuine EOS cases "
                    "will be missed.\n\n"
                    "Best when: Specialist Lab lab capacity is constrained and unnecessary trips are costly."
                ),
            ):
                st.session_state['threshold_slider'] = thresh_b05
                st.rerun()

        # ── Continuous slider ──────────────────────────────────────────────
        active_threshold = st.slider(
            "EOS probability threshold",
            min_value=0.01,
            max_value=0.99,
            step=0.01,
            key='threshold_slider',
            help=(
                "Slide ◀ LEFT  → lowers the threshold → more samples routed to Specialist Lab "
                "(captures more EOS cases, but increases false alarms).\n\n"
                "Slide ▶ RIGHT → raises the threshold → fewer samples routed to Specialist Lab "
                "(fewer false alarms, but some real EOS cases may be missed)."
            ),
        )

        # ── Descriptive label below slider ─────────────────────────────────
        if active_threshold <= 0.10:
            thr_label = (
                "🔴 Very High Recall — Almost every case is sent to Specialist Lab. "
                "Maximum EOS sensitivity, but many Non-EOS samples will follow."
            )
            thr_color = "#e74c3c"
        elif active_threshold <= 0.25:
            thr_label = (
                "🟠 High Recall — Most borderline EOS cases are caught. "
                "Some Non-EOS samples will reach Specialist Lab as false positives."
            )
            thr_color = "#e67e22"
        elif active_threshold <= 0.45:
            thr_label = (
                "🟡 Recall-favoured — Good EOS sensitivity. "
                "A proportion of Non-EOS cases accepted as false positives."
            )
            thr_color = "#f39c12"
        elif active_threshold <= 0.55:
            thr_label = (
                "⚖️ Balanced — Equal trade-off between catching EOS and avoiding false alarms. "
                "Recommended default starting point."
            )
            thr_color = "#3498db"
        elif active_threshold <= 0.75:
            thr_label = (
                "🟢 Precision-favoured — Fewer false alarms. "
                "Some borderline EOS cases may be missed."
            )
            thr_color = "#27ae60"
        else:
            thr_label = (
                "🔵 High Precision — Only very confident EOS predictions reach Specialist Lab. "
                "Risk of missing weaker EOS cases."
            )
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
        st.markdown("**Model info**")
        st.markdown(f"- Name: `{meta.get('model_name','RF-SMOTE')}`")
        st.markdown(f"- Strategy: `{meta.get('data_strategy','smote')}`")
        st.markdown(f"- Macro F1: **{meta.get('macro_f1',0):.4f}**")
        st.markdown(f"- EOS-Precision: **{meta.get('eos_precision',0):.4f}**")
        st.markdown(f"- EOS-Recall: **{meta.get('eos_recall',0):.4f}**")
        st.markdown(f"- Train rows: {meta.get('train_rows',0):,}")
        st.markdown(f"- Test rows: {meta.get('test_rows',0):,}")
        st.markdown(f"- TF-IDF features: {meta.get('tfidf_features',3000):,}")
        st.markdown("---")
        st.markdown(
            f"<div style='font-size:0.77rem;color:#6c757d;'>"
            f"Active threshold: <b style='color:#1565C0;'>{active_threshold:.2f}</b></div>",
            unsafe_allow_html=True,
        )

    # ── Title ─────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="title-row">🤖 FA Failure Classifier</div>',
        unsafe_allow_html=True,
    )

    # ── Stats bar ─────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="stats-bar">'
        f'<b>Model:</b> {meta.get("model_name","RF-SMOTE")} &nbsp;|&nbsp; '
        f'<b>Accuracy:</b> {meta.get("accuracy",0)*100:.1f}% &nbsp;|&nbsp; '
        f'<b>Macro F1:</b> {meta.get("macro_f1",0):.4f} &nbsp;|&nbsp; '
        f'<b>EOS F1:</b> {meta.get("eos_f1",0):.4f} &nbsp;|&nbsp; '
        f'<b>EOS Prec:</b> {meta.get("eos_precision",0):.4f} &nbsp;|&nbsp; '
        f'<b>EOS Rec:</b> {meta.get("eos_recall",0):.4f} &nbsp;|&nbsp; '
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
        st.markdown('<div class="predict-btn-col">', unsafe_allow_html=True)
        predict_clicked = st.button("🔮 Predict", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    route_placeholder = col_route.empty()

    # ── Prediction logic ──────────────────────────────────────────────────
    if predict_clicked:
        if not user_input.strip():
            st.warning("⚠️ Please enter a failure description.")
            st.session_state.prediction = None
        else:
            cleaned = clean_text(user_input)
            X_vec   = tfidf.transform([cleaned])
            if scaler is not None:
                X_vec = scaler.transform(X_vec)

            proba    = model.predict_proba(X_vec)[0]
            eos_prob = float(proba[eos_idx])

            # Apply threshold
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
    st.markdown(
        f'<div style="text-align:right;font-size:0.72rem;color:#999;'
        f'margin-top:-0.3rem;margin-bottom:0.2rem;">'
        f'🕐 Predicted at <b>{pred["timestamp"]}</b> &nbsp;|&nbsp; '
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
                '<div style="font-size:0.70rem;color:#6c757d;margin-top:4px;'
                'padding:0.35rem 0.45rem;background:#f8f9fa;border-radius:4px;'
                'line-height:1.45;">'
                '<b>EOS freq %</b> — how often this keyword appeared in <i>EOS-labelled</i> '
                'training samples (e.g. 25% means 1 in 4 EOS cases contained it).<br>'
                '<b>Non-EOS freq %</b> — same count for Non-EOS samples.<br>'
                'These are <b>independent dataset frequencies</b>, not probability '
                'contributions — they do <b>not</b> sum to the EOS probability above. '
                'The model score (83.3%) is computed by 200 decision trees voting across '
                '3,000 features simultaneously.'
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
        else:
            st.caption("No top EOS model features found in this input.")


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
