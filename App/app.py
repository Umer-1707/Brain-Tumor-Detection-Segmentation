import os
import base64
import tempfile
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

from App.inference import run_inference


MIDNIGHT       = "#08141B"
ELITE_TEAL     = "#132F37"
ATLANTIC_DEEP  = "#2A4F58"
TORNADO        = "#51727A"
TRANQUIL_AQUA  = "#7B99A0"
JULIET_BLUE    = "#A6C0C6"
WHITE_SEA      = "#D7E6EA"


st.set_page_config(
    page_title="Brain Tumor Detection & Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

    <style>
        :root {{
            --midnight: {MIDNIGHT};
            --elite-teal: {ELITE_TEAL};
            --atlantic-deep: {ATLANTIC_DEEP};
            --tornado: {TORNADO};
            --tranquil-aqua: {TRANQUIL_AQUA};
            --juliet-blue: {JULIET_BLUE};
            --white-sea: {WHITE_SEA};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        #MainMenu, footer, header {{ visibility: hidden; }}

        .stApp {{
            background:
                radial-gradient(circle at 50% -10%, var(--elite-teal) 0%, var(--midnight) 55%) fixed,
                repeating-linear-gradient(0deg, rgba(122,153,160,0.035) 0px, rgba(122,153,160,0.035) 1px, transparent 1px, transparent 42px),
                repeating-linear-gradient(90deg, rgba(122,153,160,0.035) 0px, rgba(122,153,160,0.035) 1px, transparent 1px, transparent 42px);
            color: var(--juliet-blue);
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--midnight) 0%, #0a1a22 100%);
            border-right: 1px solid var(--atlantic-deep);
        }}

        .app-eyebrow {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            letter-spacing: 0.22em;
            color: var(--tranquil-aqua);
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }}
        .app-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 2.6rem;
            color: var(--white-sea);
            margin: 0;
            line-height: 1.15;
        }}
        .app-subtitle {{
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
            color: var(--juliet-blue);
            margin-top: 0.5rem;
            max-width: 640px;
        }}
        .app-divider {{
            height: 1px;
            width: 100%;
            margin: 1.6rem 0 2rem 0;
            background: linear-gradient(90deg, var(--tranquil-aqua) 0%, var(--atlantic-deep) 40%, transparent 100%);
        }}

        .step-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--tranquil-aqua);
            margin-bottom: 0.5rem;
        }}

        .card {{
            background: var(--elite-teal);
            border: 1px solid var(--atlantic-deep);
            border-radius: 14px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1rem;
        }}
        .card h4 {{
            font-family: 'Space Grotesk', sans-serif;
            color: var(--white-sea);
            margin: 0 0 0.8rem 0;
            font-size: 1.05rem;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            background: var(--elite-teal) !important;
            border: 1.5px dashed var(--atlantic-deep) !important;
            border-radius: 14px !important;
        }}
        [data-testid="stFileUploaderDropzone"] * {{
            color: var(--juliet-blue) !important;
        }}
        [data-testid="stFileUploader"] section button {{
            background: var(--atlantic-deep) !important;
            color: var(--white-sea) !important;
            border: 1px solid var(--tranquil-aqua) !important;
            border-radius: 8px !important;
        }}

        .scan-frame {{
            position: relative;
            overflow: hidden;
            border-radius: 14px;
            border: 1px solid var(--atlantic-deep);
            background: var(--midnight);
            line-height: 0;
        }}
        .scan-frame img {{
            width: 100%;
            display: block;
            opacity: 0.94;
        }}
        .scan-line {{
            position: absolute;
            left: 0; right: 0;
            height: 2px;
            top: 0;
            background: linear-gradient(90deg, transparent, var(--white-sea) 50%, transparent);
            box-shadow: 0 0 14px 2px var(--tranquil-aqua);
            animation: scan-sweep 2.4s ease-in-out infinite;
        }}
        @keyframes scan-sweep {{
            0%   {{ top: 0%; }}
            50%  {{ top: 100%; }}
            100% {{ top: 0%; }}
        }}
        .scan-tag {{
            position: absolute;
            top: 10px; right: 10px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.1em;
            color: var(--white-sea);
            background: rgba(8, 20, 27, 0.75);
            border: 1px solid var(--tranquil-aqua);
            padding: 3px 8px;
            border-radius: 999px;
        }}
        .frame-caption {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--tornado);
            margin-top: 0.5rem;
            letter-spacing: 0.04em;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 5px 12px;
            border-radius: 999px;
            border: 1px solid var(--tranquil-aqua);
            color: var(--white-sea);
            background: rgba(123, 153, 160, 0.12);
        }}
        .badge.clear {{
            border-color: var(--tornado);
            color: var(--juliet-blue);
            background: rgba(81, 114, 122, 0.12);
        }}
        .diagnosis-label {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.7rem;
            font-weight: 600;
            color: var(--white-sea);
            margin: 0.6rem 0 0.2rem 0;
            text-transform: capitalize;
        }}

        .conf-row {{
            display: flex;
            justify-content: space-between;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            color: var(--juliet-blue);
            margin-top: 1rem;
            margin-bottom: 0.35rem;
        }}
        .conf-value {{
            color: var(--white-sea);
            font-weight: 600;
        }}
        .conf-track {{
            width: 100%;
            height: 8px;
            border-radius: 999px;
            background: var(--midnight);
            border: 1px solid var(--atlantic-deep);
            overflow: hidden;
        }}
        .conf-fill {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--atlantic-deep), var(--tranquil-aqua), var(--white-sea));
        }}

        .sb-card {{
            background: var(--elite-teal);
            border: 1px solid var(--atlantic-deep);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
            font-size: 0.85rem;
            color: var(--juliet-blue);
        }}
        .sb-title {{
            font-family: 'Space Grotesk', sans-serif;
            color: var(--white-sea);
            font-size: 0.95rem;
            margin-bottom: 0.3rem;
        }}

        p, span, label, .stMarkdown {{ color: var(--juliet-blue); }}
        .footnote {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: var(--tornado);
            margin-top: 2.5rem;
            text-align: center;
            letter-spacing: 0.05em;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("<div class='app-eyebrow'>SYSTEM</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sb-title' style='font-size:1.15rem;'>Scan Console</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sb-card">
            <div class="sb-title">Classification model</div>
            Flags whether a tumor is present in the uploaded MRI slice.
        </div>
        <div class="sb-card">
            <div class="sb-title">Segmentation model</div>
            Traces the tumor boundary when one is detected.
        </div>
        <div class="sb-card">
            <div class="sb-title">Disclaimer</div>
            Research / educational tool only. Not a substitute for
            professional radiological diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="app-eyebrow">MRI DIAGNOSTIC PIPELINE</div>
    <div class="app-title">🧠 Brain Tumor Detection &amp; Segmentation</div>
    <div class="app-subtitle">
        Upload an MRI slice to run it through the classification and
        segmentation models — the result includes a diagnosis, a
        confidence score, and a traced tumor boundary when applicable.
    </div>
    <div class="app-divider"></div>
    """,
    unsafe_allow_html=True,
)


st.markdown("<div class='step-label'>STEP 1 — UPLOAD SCAN</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)


def build_overlay(original_path: str, mask_array: np.ndarray, tint_hex: str, alpha: int = 120) -> Image.Image:
    tint = tuple(int(tint_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))

    base = Image.open(original_path).convert("RGBA")

    mask = mask_array
    if mask.dtype != np.uint8:
        if mask.max() <= 1:
            mask = (mask * 255).astype("uint8")
        else:
            mask = mask.astype("uint8")

    mask_img = Image.fromarray(mask).convert("L").resize(base.size)

    overlay = Image.new("RGBA", base.size, tint + (0,))
    alpha_channel = mask_img.point(lambda p: alpha if p > 127 else 0)
    overlay.putalpha(alpha_channel)

    return Image.alpha_composite(base, overlay)


if uploaded_file is not None:

    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_image_path = temp_file.name

    img_b64 = base64.b64encode(uploaded_file.getvalue()).decode()
    mime = uploaded_file.type or "image/png"

    st.markdown("<div class='step-label'>STEP 2 — SCANNING</div>", unsafe_allow_html=True)

    scan_placeholder = st.empty()
    scan_placeholder.markdown(
        f"""
        <div class="scan-frame">
            <span class="scan-tag">ANALYZING</span>
            <img src="data:{mime};base64,{img_b64}" />
            <div class="scan-line"></div>
        </div>
        <div class="frame-caption">running classification + segmentation models...</div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner(""):
        result = run_inference(temp_image_path)

    scan_placeholder.markdown(
        f"""
        <div class="scan-frame">
            <span class="scan-tag">SOURCE SCAN</span>
            <img src="data:{mime};base64,{img_b64}" />
        </div>
        <div class="frame-caption">{uploaded_file.name}</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='step-label' style='margin-top:2rem;'>STEP 3 — CLASSIFICATION RESULT</div>", unsafe_allow_html=True)

    prediction = result["prediction"]
    confidence = result["confidence"]
    tumor_detected = result["segmentation"] is not None

    badge_class = "" if tumor_detected else "clear"
    badge_text = "Tumor Detected" if tumor_detected else "No Tumor Detected"
    conf_pct = max(0.0, min(1.0, confidence)) * 100

    st.markdown(
        f"""
        <div class="card">
            <span class="badge {badge_class}">● {badge_text}</span>
            <div class="diagnosis-label">{prediction}</div>
            <div class="conf-row">
                <span>CONFIDENCE</span>
                <span class="conf-value">{confidence:.2%}</span>
            </div>
            <div class="conf-track">
                <div class="conf-fill" style="width:{conf_pct:.1f}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='step-label'>STEP 4 — SEGMENTATION</div>", unsafe_allow_html=True)

    if tumor_detected:

        mask_array = result["segmentation"]
        mask_display = (mask_array * 255).astype("uint8") if mask_array.max() <= 1 else mask_array.astype("uint8")

        show_overlay = st.toggle("Show tumor overlay on original scan", value=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("<div class='card'><h4>Predicted Tumor Mask</h4>", unsafe_allow_html=True)
            st.image(mask_display, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown("<div class='card'><h4>Overlay on Scan</h4>", unsafe_allow_html=True)
            if show_overlay:
                overlay_img = build_overlay(temp_image_path, mask_array, TRANQUIL_AQUA)
                st.image(overlay_img, width="stretch")
            else:
                st.image(uploaded_file, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        buf = BytesIO()
        Image.fromarray(mask_display).save(buf, format="PNG")
        st.download_button(
            "Download segmentation mask",
            data=buf.getvalue(),
            file_name=f"mask_{os.path.splitext(uploaded_file.name)[0]}.png",
            mime="image/png",
        )

    else:
        st.markdown(
            """
            <div class="card">
                <span class="badge clear">● Clear</span>
                <p style="margin-top:0.8rem; margin-bottom:0;">
                    No tumor was detected, so no segmentation mask was generated.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    os.remove(temp_image_path)

else:
    st.markdown(
        """
        <div class="card" style="text-align:center; padding: 2.4rem 1.5rem;">
            <div style="font-size:1.6rem;">🧠</div>
            <p style="margin-top:0.6rem; margin-bottom:0;">
                Waiting for an MRI scan — upload a JPG or PNG to begin.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='footnote'>BRAIN TUMOR DETECTION & SEGMENTATION · RESEARCH USE ONLY</div>",
    unsafe_allow_html=True,
)
