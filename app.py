import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Waste Classifier",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    "cardboard": "#1AB2F2",
    "glass":     "#2D5C3F",
    "metal":     "#3F7A4E",
    "paper":     "#5A9B68",
    "plastic":   "#3F7A4E",
    "trash":     "#1AB2F2",
    "primary":   "#1AB2F2",
    "secondary": "#2D5C3F",
    "accent":    "#3F7A4E",
    "light":     "#5A9B68",
    "background":"#FFFFFF",
    "text":      "#2C3E50",
    "border":    "#E0E0E0",
}

# ── Per-model performance data ─────────────────────────────────────────────────
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

MODEL_STATS = {
    "exp3": {
        "label":    "Exp 3  —  CW + Fine-tuning",
        "subtitle": "Class Weights + Fine-tuning (30 layers, lr=1e-5)",
        "accuracy": 87.17,
        "loss":     0.5926,
        
        "precision": {"cardboard":0.742, "glass":0.691, "metal":0.866, "paper":0.808, "plastic":0.842, "trash":0.968},
        "recall":    {"cardboard":0.961, "glass":0.859, "metal":0.829, "paper":0.832, "plastic":0.810, "trash":0.645},
        "f1":        {"cardboard":0.838, "glass":0.766, "metal":0.847, "paper":0.820, "plastic":0.826, "trash":0.774},
    },
    "exp3v2": {
        "label":    "Exp 3 V2  —  Data Aug + CW + FT",
        "subtitle": "Offline trash augmentation + Class Weights + Fine-tuning",
        "accuracy": 89.62,
        "loss":     0.3515,
        
        "precision": {"cardboard":0.907, "glass":0.841, "metal":0.863, "paper":0.924, "plastic":0.828, "trash":1.000},
        "recall":    {"cardboard":0.925, "glass":0.881, "metal":0.932, "paper":0.924, "plastic":0.828, "trash":0.880},
        "f1":        {"cardboard":0.916, "glass":0.860, "metal":0.896, "paper":0.924, "plastic":0.828, "trash":0.936},
    },
}
# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    color: {COLORS['text']};
    background: {COLORS['background']};
}}

/* Remove Streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 2.5rem 3rem 4rem; max-width: 1200px; }}

/* Top header bar */
.app-header {{
    display: flex;
    align-items: baseline;
    gap: 1.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid {COLORS['border']};
    margin-bottom: 2.5rem;
}}
.app-title {{
    font-family: 'DM Mono', monospace;
    font-size: 1.05rem;
    font-weight: 500;
    color: {COLORS['text']};
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}
.app-subtitle {{
    font-size: 0.8rem;
    font-weight: 300;
    color: #90A4A4;
    letter-spacing: 0.02em;
}}

/* Upload zone */
.upload-zone {{
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 2.5rem 2rem;
    text-align: center;
    background: #FAFAFA;
}}
.upload-label {{
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #90A4A4;
    margin-bottom: 0.5rem;
}}

/* Model card */
.model-card {{
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 1.5rem;
    background: #FFFFFF;
    height: 100%;
}}
.model-card-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1.25rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid {COLORS['border']};
}}
.model-name {{
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {COLORS['secondary']};
}}
.model-desc {{
    font-size: 0.72rem;
    font-weight: 300;
    color: #90A4A4;
    margin-top: 0.2rem;
}}

/* Prediction result */
.pred-class {{
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 0.2rem;
}}
.pred-conf {{
    font-size: 0.78rem;
    font-weight: 300;
    color: #90A4A4;
    letter-spacing: 0.04em;
}}

/* Confidence bar */
.bar-row {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.45rem;
}}
.bar-label {{
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: {COLORS['text']};
    width: 72px;
    flex-shrink: 0;
}}
.bar-track {{
    flex: 1;
    height: 3px;
    background: #F0F0F0;
    border-radius: 2px;
    overflow: hidden;
}}
.bar-fill {{
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
}}
.bar-pct {{
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #90A4A4;
    width: 36px;
    text-align: right;
    flex-shrink: 0;
}}

/* Stats grid */
.stats-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-top: 1.25rem;
    padding-top: 1.25rem;
    border-top: 1px solid {COLORS['border']};
}}
.stat-cell {{
    text-align: center;
}}
.stat-value {{
    font-family: 'DM Mono', monospace;
    font-size: 1.1rem;
    font-weight: 500;
    color: {COLORS['text']};
    line-height: 1;
}}
.stat-label {{
    font-size: 0.65rem;
    font-weight: 300;
    color: #90A4A4;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}}

/* Per-class metrics table */
.metrics-section {{
    margin-top: 1.25rem;
    padding-top: 1.25rem;
    border-top: 1px solid {COLORS['border']};
}}
.metrics-title {{
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #90A4A4;
    margin-bottom: 0.75rem;
}}
.metrics-row {{
    display: grid;
    grid-template-columns: 72px 1fr 1fr 1fr;
    gap: 0.4rem;
    align-items: center;
    padding: 0.3rem 0;
    border-bottom: 1px solid #F5F5F5;
    font-size: 0.7rem;
}}
.metrics-header {{
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #AAAAAA;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid {COLORS['border']};
    margin-bottom: 0.2rem;
}}
.metrics-cls {{
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: {COLORS['text']};
}}
.metrics-val {{
    font-family: 'DM Mono', monospace;
    text-align: right;
    color: {COLORS['text']};
}}
.metrics-val.highlight {{
    color: {COLORS['primary']};
    font-weight: 500;
}}

/* Section divider */
.section-label {{
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #90A4A4;
    margin-bottom: 1rem;
}}

/* Streamlit overrides */
.stFileUploader > div {{
    border: none !important;
    background: transparent !important;
}}
div[data-testid="stFileUploader"] {{
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    background: #FAFAFA;
    padding: 1rem;
}}
div[data-testid="stFileUploader"] label {{
    font-size: 0.7rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #90A4A4 !important;
}}
.stButton > button {{
    background: {COLORS['secondary']} !important;
    color: white !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 500 !important;
}}
.stButton > button:hover {{
    background: {COLORS['accent']} !important;
}}
</style>
""", unsafe_allow_html=True)


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        m1 = tf.keras.models.load_model("./model/best_model.keras")
        m2 = tf.keras.models.load_model("./model/v2_best_model.keras")
        return m1, m2
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None


@st.cache_resource(show_spinner=False)
def load_class_names():
    """
    Load class names in the EXACT order used by image_dataset_from_directory.
    That function sorts folder names alphabetically — so we do the same.
    We also try the saved .npy as a cross-check.
    """
    import os

    # Ground truth: sorted subfolder names in dataset/
    dataset_candidates = ["./dataset", "../dataset"]
    folder_names = None
    for path in dataset_candidates:
        if os.path.isdir(path):
            folder_names = sorted([
                d for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d))
                and not d.startswith(".")
            ])
            break

    # Also load .npy as reference
    npy_names = None
    try:
        npy_names = np.load("./model/class_names.npy", allow_pickle=True).tolist()
    except Exception:
        pass

    # Use folder scan if available (most reliable), else npy, else hardcoded
    if folder_names and len(folder_names) == 6:
        names = folder_names
    elif npy_names:
        names = npy_names
    else:
        names = sorted(["cardboard", "glass", "metal", "paper", "plastic", "trash"])

    return names, folder_names, npy_names


def preprocess(img: Image.Image) -> np.ndarray:
    """
    The model already contains preprocess_input as a Lambda layer internally
    (baked in during training with the Functional API).
    So we only resize and convert to float — do NOT call preprocess_input here,
    that would normalize twice and corrupt the input.
    """
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)   # values stay in [0, 255]
    return np.expand_dims(arr, 0)


def predict(model, arr, class_names):
    preds = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(preds))
    return class_names[idx], float(preds[idx]), preds.tolist()


# ── HTML helpers ───────────────────────────────────────────────────────────────
def confidence_bars(probs, predicted_class):
    rows = ""
    for cls, prob in zip(CLASS_NAMES, probs):
        pct   = prob * 100
        color = COLORS["primary"] if cls == predicted_class else COLORS["border"]
        rows += f"""
        <div class="bar-row">
          <div class="bar-label">{cls}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
          </div>
          <div class="bar-pct">{pct:.1f}%</div>
        </div>"""
    return rows


def metrics_table(stats):
    rows = ""
    for cls in CLASS_NAMES:
        p = stats["precision"][cls]
        r = stats["recall"][cls]
        f = stats["f1"][cls]
        hi_p = ' highlight' if p == max(stats["precision"].values()) else ''
        hi_r = ' highlight' if cls == "trash" else ''
        hi_f = ' highlight' if f == max(stats["f1"].values()) else ''
        rows += f"""
        <div class="metrics-row">
          <div class="metrics-cls">{cls}</div>
          <div class="metrics-val{hi_p}">{p:.3f}</div>
          <div class="metrics-val{hi_r}">{r:.3f}</div>
          <div class="metrics-val{hi_f}">{f:.3f}</div>
        </div>"""
    return rows


CARD_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: transparent; font-family: 'DM Sans', sans-serif; color: {COLORS['text']}; }}
.model-card {{ border: 1px solid {COLORS['border']}; border-radius: 4px; padding: 1.25rem; background: #fff; }}
.model-card-header {{ display: flex; justify-content: space-between; align-items: flex-start;
    margin-bottom: 1rem; padding-bottom: 0.85rem; border-bottom: 1px solid {COLORS['border']}; }}
.model-name {{ font-family: 'DM Mono', monospace; font-size: 0.68rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.06em; color: {COLORS['secondary']}; }}
.model-desc {{ font-size: 0.68rem; font-weight: 300; color: #90A4A4; margin-top: 0.15rem; }}
.pred-class {{ font-family: 'DM Mono', monospace; font-size: 1.75rem; font-weight: 500;
    letter-spacing: -0.02em; line-height: 1; margin-bottom: 0.15rem; }}
.pred-conf {{ font-size: 0.72rem; font-weight: 300; color: #90A4A4; letter-spacing: 0.04em; margin-bottom: 1rem; }}
.bar-row {{ display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem; }}
.bar-label {{ font-family: 'DM Mono', monospace; font-size: 0.65rem; color: {COLORS['text']}; width: 66px; flex-shrink: 0; }}
.bar-track {{ flex: 1; height: 3px; background: #F0F0F0; border-radius: 2px; overflow: hidden; }}
.bar-fill {{ height: 100%; border-radius: 2px; }}
.bar-pct {{ font-family: 'DM Mono', monospace; font-size: 0.62rem; color: #90A4A4; width: 34px; text-align: right; flex-shrink: 0; }}
.dist-label {{ font-size: 0.62rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase;
    color: #90A4A4; margin-bottom: 0.6rem; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem;
    margin-top: 1rem; padding-top: 1rem; border-top: 1px solid {COLORS['border']}; }}
.stat-cell {{ text-align: center; }}
.stat-value {{ font-family: 'DM Mono', monospace; font-size: 1rem; font-weight: 500;
    color: {COLORS['text']}; line-height: 1; }}
.stat-label {{ font-size: 0.6rem; font-weight: 300; color: #90A4A4; letter-spacing: 0.05em;
    text-transform: uppercase; margin-top: 0.2rem; }}
.metrics-section {{ margin-top: 1rem; padding-top: 1rem; border-top: 1px solid {COLORS['border']}; }}
.metrics-title {{ font-size: 0.62rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase;
    color: #90A4A4; margin-bottom: 0.6rem; }}
.metrics-row {{ display: grid; grid-template-columns: 72px 1fr 1fr 1fr; gap: 0.3rem;
    align-items: center; padding: 0.28rem 0; border-bottom: 1px solid #F5F5F5; font-size: 0.68rem; }}
.metrics-header-row {{ display: grid; grid-template-columns: 72px 1fr 1fr 1fr; gap: 0.3rem;
    align-items: center; padding-bottom: 0.35rem; border-bottom: 1px solid {COLORS['border']};
    margin-bottom: 0.15rem; font-family: 'DM Mono', monospace; font-size: 0.6rem;
    font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase; color: #AAAAAA; }}
.metrics-cls {{ font-family: 'DM Mono', monospace; font-size: 0.65rem; color: {COLORS['text']}; }}
.metrics-val {{ font-family: 'DM Mono', monospace; text-align: right; color: {COLORS['text']}; }}
.metrics-val.hl {{ color: {COLORS['primary']}; font-weight: 500; }}
</style>
"""


def render_card(col, stats, pred_class=None, pred_conf=None, probs=None, class_names=None):
    if class_names is None:
        class_names = CLASS_NAMES
    pred_color = COLORS.get(pred_class, COLORS["primary"]) if pred_class else COLORS["border"]

    pred_html = ""
    bars_html = ""
    if pred_class and probs is not None:
        pred_html = f"""
        <div class="pred-class" style="color:{pred_color};">{pred_class}</div>
        <div class="pred-conf">{pred_conf*100:.2f}% confidence</div>"""

        bar_rows = ""
        for cls, prob in zip(class_names, probs):
            pct   = prob * 100
            color = COLORS["primary"] if cls == pred_class else COLORS["border"]
            bar_rows += f"""
            <div class="bar-row">
              <div class="bar-label">{cls}</div>
              <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div></div>
              <div class="bar-pct">{pct:.1f}%</div>
            </div>"""
        bars_html = f'<div class="dist-label" style="margin-top:0.75rem;">Probability distribution</div>{bar_rows}'

    metric_rows = ""
    for cls in CLASS_NAMES:
        p    = stats["precision"][cls]
        r    = stats["recall"][cls]
        f    = stats["f1"][cls]
        hl_p = " hl" if p == max(stats["precision"].values()) else ""
        hl_r = " hl" if cls == "trash" else ""
        hl_f = " hl" if f == max(stats["f1"].values()) else ""
        metric_rows += f"""
        <div class="metrics-row">
          <div class="metrics-cls">{cls}</div>
          <div class="metrics-val{hl_p}">{p:.3f}</div>
          <div class="metrics-val{hl_r}">{r:.3f}</div>
          <div class="metrics-val{hl_f}">{f:.3f}</div>
        </div>"""

    # Estimate height: base + prediction section if present
    height = 520 if not pred_class else 780

    html = f"""
    {CARD_CSS}
    <div class="model-card">
      <div class="model-card-header">
        <div>
          <div class="model-name">{stats['label']}</div>
          <div class="model-desc">{stats['subtitle']}</div>
        </div>
      </div>
      {pred_html}
      {bars_html}
      <div class="stats-grid">
        <div class="stat-cell">
          <div class="stat-value">{stats['accuracy']:.1f}%</div>
          <div class="stat-label">Test accuracy</div>
        </div>
        <div class="stat-cell">
          <div class="stat-value">{stats['loss']:.4f}</div>
          <div class="stat-label">Test loss</div>
        </div>
      </div>
      <div class="metrics-section">
        <div class="metrics-title">Per-class metrics</div>
        <div class="metrics-header-row">
          <div>Class</div>
          <div style="text-align:right">Prec.</div>
          <div style="text-align:right">Recall</div>
          <div style="text-align:right">F1</div>
        </div>
        {metric_rows}
      </div>
    </div>
    """
    with col:
        components.html(html, height=height, scrolling=False)


# ── Layout ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">Waste Classifier</div>
  <div class="app-subtitle">MobileNetV2 — Transfer Learning — 6 classes</div>
</div>
""", unsafe_allow_html=True)

# Two-column layout: upload left, results right
col_upload, col_gap, col_results = st.columns([1, 0.08, 2.2])

with col_upload:
    st.markdown('<div class="section-label">Input image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop an image or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True)
        st.markdown(f"""
        <div style="margin-top:0.75rem;font-family:'DM Mono',monospace;font-size:0.65rem;
                    color:#90A4A4;letter-spacing:0.04em;">
          {uploaded.name}<br>
          {img.size[0]} x {img.size[1]} px
        </div>""", unsafe_allow_html=True)

with col_results:
    st.markdown('<div class="section-label">Model comparison</div>', unsafe_allow_html=True)

    model_exp3, model_exp3v2 = load_models()
    class_names_loaded, folder_names, npy_names = load_class_names()
    models_ok = model_exp3 is not None and model_exp3v2 is not None

    # ── Debug expander (remove before final demo) ─────────────────────────
    with st.expander("Debug info", expanded=False):
        st.code(
            f"Class names used       : {class_names_loaded}\n"
            f"From dataset folders   : {folder_names}\n"
            f"From class_names.npy   : {npy_names}\n"
            f"Hardcoded CLASS_NAMES  : {CLASS_NAMES}\n"
            f"Models loaded          : {models_ok}"
        )

    if not models_ok:
        c1, c2 = st.columns(2)
        render_card(c1, MODEL_STATS["exp3"])
        render_card(c2, MODEL_STATS["exp3v2"])
        st.markdown("""
        <div style="margin-top:1rem;padding:0.75rem 1rem;border:1px solid #F0F0F0;border-radius:4px;
                    font-size:0.72rem;color:#90A4A4;font-family:'DM Mono',monospace;">
          Models not found at ./model/. Upload an image once models are available.
        </div>""", unsafe_allow_html=True)

    elif not uploaded:
        c1, c2 = st.columns(2)
        render_card(c1, MODEL_STATS["exp3"])
        render_card(c2, MODEL_STATS["exp3v2"])

    else:
        arr = preprocess(img)

        pred1_cls, pred1_conf, probs1 = predict(model_exp3,   arr, class_names_loaded)
        pred2_cls, pred2_conf, probs2 = predict(model_exp3v2, arr, class_names_loaded)

        # Debug: show raw probabilities
        with st.expander("Raw probabilities", expanded=False):
            st.write("**Exp3:**", {class_names_loaded[i]: f"{p:.4f}" for i, p in enumerate(probs1)})
            st.write("**Exp3 V2:**", {class_names_loaded[i]: f"{p:.4f}" for i, p in enumerate(probs2)})

        c1, c2 = st.columns(2)
        render_card(c1, MODEL_STATS["exp3"],   pred1_cls, pred1_conf, probs1, class_names_loaded)
        render_card(c2, MODEL_STATS["exp3v2"], pred2_cls, pred2_conf, probs2, class_names_loaded)

        # Agreement banner
        if pred1_cls == pred2_cls:
            banner_color = COLORS["accent"]
            banner_msg   = f"Both models agree &mdash; <strong>{pred1_cls}</strong>"
        else:
            banner_color = "#E0A030"
            banner_msg   = f"Models disagree &mdash; Exp3: <strong>{pred1_cls}</strong> &nbsp;|&nbsp; V2: <strong>{pred2_cls}</strong>"

        st.markdown(f"""
        <div style="margin-top:0.75rem;padding:0.6rem 1rem;
                    border-left:3px solid {banner_color};
                    font-size:0.72rem;color:{COLORS['text']};
                    font-family:'DM Sans',sans-serif;background:#FAFAFA;">
          {banner_msg}
        </div>""", unsafe_allow_html=True)