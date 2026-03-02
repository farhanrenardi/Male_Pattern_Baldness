import io
import os
import time
from typing import Optional

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input


# Minimal CONFIG extracted/derived from notebook
CONFIG = {
    "input": {"image_size": (300, 300)},
    "dataset": {"allowed_classes": [str(i) for i in range(1, 8)]},
    "model": {"backbone": "EfficientNetB3", "pretrained": True, "dropout": 0.2},
    "artifacts": {"default_weights": os.path.join("EXP-KF-004_artifacts", "best.weights.h5")},
    "rejection": {"min_confidence": 0.35, "quality_gate": {"blur_laplacian_threshold": 100.0}},
}


def build_model(config: dict) -> tf.keras.Model:
    h, w = config["input"]["image_size"]
    inputs = layers.Input(shape=(h, w, 3), name="input_image")
    backbone = EfficientNetB3(include_top=False, weights="imagenet" if config["model"]["pretrained"] else None,
                              input_tensor=inputs)
    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    if config["model"].get("dropout", 0) > 0:
        x = layers.Dropout(config["model"]["dropout"])(x)
    num_classes = len(config["dataset"]["allowed_classes"])
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


class MPHLPredictor:
    def __init__(self, model: tf.keras.Model, config: dict):
        self.model = model
        self.config = config

    def check_blur(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(fm)

    def predict_single(self, image: np.ndarray, check_quality: bool = True) -> dict:
        # image: HxWx3 RGB uint8 or float
        img_resized = cv2.resize(image, self.config["input"]["image_size"][::-1])
        img = img_resized.astype(np.float32)
        x = preprocess_input(img)
        x = np.expand_dims(x, 0)
        preds = self.model.predict(x)
        class_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        class_name = self.config["dataset"]["allowed_classes"][class_idx]

        blur_score = self.check_blur(image)
        rejection_reason = None
        if check_quality:
            if confidence < self.config["rejection"]["min_confidence"]:
                rejection_reason = "low_confidence"
            elif blur_score < self.config["rejection"]["quality_gate"]["blur_laplacian_threshold"]:
                rejection_reason = "too_blurry"

        return {
            "output": int(class_name),
            "class_idx": class_idx,
            "class_name": class_name,
            "confidence": confidence,
            "rejection_reason": rejection_reason,
            "blur_score": blur_score,
        }


LEVEL_COLORS = {
    1: (48, 209, 88),
    2: (52, 199, 89),
    3: (255, 214, 10),
    4: (255, 159, 10),
    5: (255, 107, 53),
    6: (255, 55, 95),
    7: (255, 45, 85),
}

NORWOOD_DESC = {
    1: "No significant hair loss",
    2: "Slight temple recession",
    3: "Deeper recession",
    4: "Vertex thinning",
    5: "Advanced vertex loss",
    6: "Vertex areas merge",
    7: "Extensive hair loss",
}


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        # macOS system fonts
        "/System/Library/Fonts/SFNSDisplay-Regular.otf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        # Windows fonts
        r"C:\\Windows\\Fonts\\arial.ttf",
        r"C:\\Windows\\Fonts\\calibri.ttf",
        r"C:\\Windows\\Fonts\\seguisym.ttf",
        # Common Linux fonts
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple:
    try:
        b = draw.textbbox((0, 0), text, font=font)
        return b[2] - b[0], b[3] - b[1]
    except Exception:
        pass
    try:
        return font.getsize(text)
    except Exception:
        return len(text) * 8, 12


def draw_label_on_image(pil_img: Image.Image, label: str, confidence: float, rejection: Optional[str]) -> Image.Image:
    img = pil_img.convert("RGB")
    img_w, img_h = img.size

    level = int(label)
    accent = LEVEL_COLORS.get(level, (10, 132, 255))

    panel_h = max(int(img_h * 0.16), 64)
    strip_h = max(3, panel_h // 18)

    out = Image.new("RGB", (img_w, img_h + panel_h), (15, 15, 17))
    out.paste(img, (0, 0))

    draw = ImageDraw.Draw(out)

    # dark panel
    draw.rectangle([(0, img_h), (img_w, img_h + panel_h)], fill=(15, 15, 17))
    # accent strip
    draw.rectangle([(0, img_h), (img_w, img_h + strip_h)], fill=accent)

    pad = int(panel_h * 0.18)
    big_size  = int(panel_h * 0.40)
    small_size = int(panel_h * 0.20)
    big_font   = _load_font(big_size)
    small_font = _load_font(small_size)

    text_y = img_h + strip_h + pad // 2

    # Level label (left, accent colour)
    level_text = f"Level {label}"
    draw.text((pad, text_y), level_text, font=big_font, fill=(*accent, 255))

    # Confidence (right, muted white)
    conf_text = f"{confidence * 100:.0f}%"
    cw, _ = _text_size(draw, conf_text, big_font)
    draw.text((img_w - cw - pad, text_y), conf_text, font=big_font, fill=(200, 200, 200, 255))

    # Sub-description
    if rejection:
        sub_text  = f"⚠  {rejection.replace('_', ' ').title()}"
        sub_color = (255, 159, 10, 255)
    else:
        sub_text  = NORWOOD_DESC.get(level, "")
        sub_color = (110, 110, 120, 255)

    _, bh = _text_size(draw, level_text, big_font)
    draw.text((pad, text_y + bh + 4), sub_text, font=small_font, fill=sub_color)

    return out


########## Streamlit UI ##########

st.set_page_config(page_title="Baldness Detector", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
/* ── Typography – Apple system font stack ── */
html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    letter-spacing: -0.01em;
}

/* ── Page background ── */
.stApp { background: #0d0d0f; }

/* ── Header ── */
.app-header { padding: 2.2rem 0 0.4rem 0; }
.app-header h1 {
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #f5f5f7;
    margin: 0 0 0.25rem 0;
    line-height: 1.1;
}
.app-header p { font-size: 0.88rem; color: #86868b; margin: 0; font-weight: 400; }
.accent-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #0a84ff;
    margin-right: 7px;
    vertical-align: middle;
    position: relative; top: -2px;
}

/* ── Section label ── */
.section-label {
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #48484a;
    margin: 0 0 0.65rem 0;
}

/* ── Streamlit uploader override ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #2a2a2e !important;
    border-radius: 12px !important;
    background: #18181c !important;
    transition: border-color 200ms ease;
}
[data-testid="stFileUploader"]:hover { border-color: #0a84ff !important; }
[data-testid="stFileUploaderDropzone"] { padding: 1.3rem 1rem !important; }

/* ── Predict button ── */
div[data-testid="stButton"] > button[kind="primary"] {
    width: 100%;
    background: #0a84ff !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.62rem 1.4rem !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    transition: background 140ms ease, transform 110ms ease, box-shadow 140ms ease !important;
    box-shadow: 0 2px 14px rgba(10,132,255,0.28) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #1a91ff !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(10,132,255,0.40) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 8px rgba(10,132,255,0.18) !important;
}

/* ── Checkbox ── */
.stCheckbox label { font-size: 0.80rem !important; color: #86868b !important; }

/* ── Result card ── */
.result-card {
    background: #18181c;
    border-radius: 14px;
    border: 1px solid #2a2a2e;
    padding: 1.1rem 1.3rem;
    margin-top: 0.75rem;
}
.result-level {
    font-size: 2.7rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    margin: 0 0 0.1rem 0;
}
.result-desc { font-size: 0.80rem; color: #86868b; margin: 0 0 0.9rem 0; }
.conf-bar-bg {
    background: #2a2a2e;
    border-radius: 6px;
    height: 5px;
    overflow: hidden;
    margin: 0.25rem 0 0.2rem 0;
}
.conf-bar-fill { height: 5px; border-radius: 6px; }
.conf-label { font-size: 0.70rem; color: #636366; margin: 0; }
.rejection-badge {
    display: inline-block;
    background: rgba(255,214,10,0.10);
    color: #ffd60a;
    border-radius: 6px;
    padding: 0.22rem 0.6rem;
    font-size: 0.74rem;
    font-weight: 500;
    margin-top: 0.75rem;
}
.meta-text { font-size: 0.68rem; color: #3a3a3c; margin-top: 0.9rem; }

/* ── Placeholder box ── */
.placeholder-box {
    background: #18181c;
    border: 1.5px dashed #2a2a2e;
    border-radius: 12px;
    min-height: 300px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #3a3a3c;
    font-size: 0.82rem;
}
.placeholder-icon { font-size: 1.9rem; margin-bottom: 0.45rem; opacity: 0.3; }

/* ── Dividers ── */
.divider { border: none; border-top: 1px solid #2a2a2e; margin: 0.9rem 0 1.2rem 0; }

/* ── Hide padding ── */
.block-container { padding-top: 0 !important; padding-bottom: 2rem !important; }
div[data-testid="column"] { padding: 0 1.1rem !important; }
.stImage > div > div > p { font-size: 0.70rem !important; color: #48484a !important; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1><span class="accent-dot"></span>Baldness Detector</h1>
  <p>Upload a head photo — the model classifies Norwood scale level 1–7</p>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
for key in ("pil_img", "result", "labeled_img", "elapsed"):
    if key not in st.session_state:
        st.session_state[key] = None

# Ensure model session state keys exist
for key in ("model_obj", "model_ready"):
    if key not in st.session_state:
        st.session_state[key] = None

# Show a friendly loading page while the default model and app resources are prepared.
# This prevents a blank page and informs users that the app is starting up.
if not st.session_state.get("model_ready"):
    load_box = st.empty()
    with load_box.container():
        st.markdown(
            """
            <div style='text-align:center;margin-top:5rem;'>
              <h2 style='margin-bottom:0.25rem;color:#ebebf0;'>Menyiapkan model dan aplikasi…</h2>
              <p style='color:#86868b;margin-top:0.1rem;'>Memuat bobot dan sumber daya. Ini mungkin memakan waktu beberapa detik.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.spinner("Memuat model dan inisialisasi…"):
            try:
                # Attempt to preload the default weights (cached by @st.cache_resource)
                st.session_state.model_obj = load_model(CONFIG["artifacts"]["default_weights"])
                st.session_state.model_ready = True
            except Exception as e:
                st.session_state.model_obj = None
                st.session_state.model_ready = False
                st.error(f"Gagal memuat model awal: {e}")
    # Remove the loading box after the attempt so the normal UI renders
    load_box.empty()

# ── Layout ───────────────────────────────────────────────────────────────────
left, right = st.columns([5, 7], gap="large")

# ── LEFT: form ──────────────────────────────────────────────────────────────
with left:
    st.markdown('<p class="section-label">Input</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        help="Upload a frontal head photo",
    )

    # Decode & cache on every new upload
    if uploaded_file is not None:
        raw = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        new_pil = Image.fromarray(rgb)
        # Reset prediction when a new file arrives
        if st.session_state.pil_img is None or new_pil.tobytes() != st.session_state.pil_img.tobytes():
            st.session_state.pil_img     = new_pil
            st.session_state.result      = None
            st.session_state.labeled_img = None
            st.session_state.elapsed     = None

    st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)

    use_custom = st.checkbox("Use custom weights file", value=False)
    custom_weights_file = None
    if use_custom:
        custom_weights_file = st.file_uploader(
            "Upload .h5 weights", type=["h5", "hdf5"], label_visibility="collapsed"
        )

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Predict", type="primary", key="predict_btn")

    if predict_btn and st.session_state.pil_img is None:
        st.markdown(
            "<p style='color:#ff453a;font-size:0.80rem;margin-top:0.45rem;'>Please upload an image first.</p>",
            unsafe_allow_html=True,
        )


# ── Model loader (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_model(weights_path: str) -> tf.keras.Model:
    model = build_model(CONFIG)
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    return model


# ── Inference ────────────────────────────────────────────────────────────────
if predict_btn and st.session_state.pil_img is not None:
    weights_path = CONFIG["artifacts"]["default_weights"]
    if use_custom and custom_weights_file is not None:
        tmp_path = os.path.join(".", "_uploaded_weights.h5")
        with open(tmp_path, "wb") as f:
            f.write(custom_weights_file.getbuffer())
        weights_path = tmp_path

    with st.spinner("Running inference…"):
        t0 = time.time()
        try:
            # If using a custom uploaded weights file, load that one. Otherwise reuse the
            # preloaded default model stored in session state when available.
            if use_custom and custom_weights_file is not None:
                model_obj = load_model(weights_path)
            else:
                model_obj = st.session_state.get("model_obj") or load_model(weights_path)
        except Exception as e:
            st.error(f"Failed loading weights: {e}")
            model_obj = build_model(CONFIG)

        predictor = MPHLPredictor(model_obj, CONFIG)
        st.session_state.result = predictor.predict_single(np.array(st.session_state.pil_img))
        st.session_state.labeled_img = draw_label_on_image(
            st.session_state.pil_img,
            st.session_state.result["class_name"],
            st.session_state.result["confidence"],
            st.session_state.result["rejection_reason"],
        )
        st.session_state.elapsed = time.time() - t0


# ── RIGHT: output ────────────────────────────────────────────────────────────
with right:
    st.markdown('<p class="section-label">Output</p>', unsafe_allow_html=True)

    if st.session_state.labeled_img is not None:
        # Show labeled image with panel
        st.image(st.session_state.labeled_img, use_container_width=True)

        res    = st.session_state.result
        lvl    = res["output"]
        conf   = res["confidence"]
        r, g, b = LEVEL_COLORS.get(lvl, (10, 132, 255))
        accent_css = f"rgb({r},{g},{b})"
        conf_pct   = int(conf * 100)
        desc       = NORWOOD_DESC.get(lvl, "")
        rejection_html = (
            f"<div class='rejection-badge'>⚠ &nbsp;{res['rejection_reason'].replace('_', ' ').title()}</div>"
            if res["rejection_reason"] else ""
        )

        st.markdown(f"""
<div class="result-card">
  <div class="result-level" style="color:{accent_css};">Level {lvl}</div>
  <div class="result-desc">{desc}</div>
  <div class="conf-bar-bg">
    <div class="conf-bar-fill" style="width:{conf_pct}%;background:{accent_css};"></div>
  </div>
  <div class="conf-label">Confidence &nbsp;<strong style="color:#ebebf0;">{conf_pct}%</strong></div>
  {rejection_html}
  <div class="meta-text">Inference {st.session_state.elapsed:.2f}s &nbsp;·&nbsp; blur {res['blur_score']:.0f}</div>
</div>
""", unsafe_allow_html=True)

    elif st.session_state.pil_img is not None:
        # Raw preview before predict
        st.image(st.session_state.pil_img, use_container_width=True)
        st.markdown(
            "<p style='font-size:0.76rem;color:#636366;margin-top:0.4rem;'>"
            "Preview — click <strong style='color:#ebebf0;'>Predict</strong> to analyse</p>",
            unsafe_allow_html=True,
        )

    else:
        # Empty state
        st.markdown("""
<div class="placeholder-box">
  <div class="placeholder-icon">🖼</div>
  <span>Result will appear here</span>
</div>
""", unsafe_allow_html=True)
