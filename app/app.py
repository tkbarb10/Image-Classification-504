import importlib.util
import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import sobel

# ---------------------------------------------------------------------------
# Load utility functions directly from source files, bypassing utils/__init__.py
# (which eagerly imports 'datasets', a heavy HuggingFace dep not needed here)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent


def _load_module(module_name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_hsv_mod = _load_module("_utils_hsv", _PROJECT_ROOT / "utils" / "hsv.py")
_color_mod = _load_module("_utils_color", _PROJECT_ROOT / "utils" / "color_metadata.py")
_edges_mod = _load_module("_utils_edges", _PROJECT_ROOT / "utils" / "gray_edges_metadata.py")

hsv = _hsv_mod.hsv
color_metadata = _color_mod.color_metadata
edges = _edges_mod.edges

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = _PROJECT_ROOT / "random_forest_model"

COLS_TO_DROP = [
    "r_max", "b_max", "g_max",
    "b_min", "r_min",
    "hue_mode", "hue_mode_threshold",
]

FEATURE_DESCRIPTIONS = {
    "r_sum": "Total red intensity across all pixels",
    "g_sum": "Total green intensity across all pixels",
    "b_sum": "Total blue intensity across all pixels",
    "r_mean": "Average red channel value",
    "g_mean": "Average green channel value",
    "b_mean": "Average blue channel value",
    "r_std": "Variation in red channel",
    "g_std": "Variation in green channel",
    "b_std": "Variation in blue channel",
    "r_median": "Median red channel value",
    "g_median": "Median green channel value",
    "b_median": "Median blue channel value",
    "g_min": "Minimum green channel value",
    "gray_mean": "Average brightness (grayscale)",
    "gray_std": "Variation in brightness",
    "edge_mean": "Average edge strength",
    "edge_std": "Variation in edge strength",
    "edge_median": "Median edge strength",
    "edge_entropy": "Complexity/randomness of edges",
    "edge_density": "Proportion of strong edges in image",
    "sat_mean": "Average color saturation",
    "sat_std": "Variation in color saturation",
    "aspect_ratio": "Image width-to-height ratio",
}

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def get_explainer(_model):
    """TreeSHAP explainer — underscore prefix prevents Streamlit from hashing the model."""
    return shap.TreeExplainer(_model)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(image_array: np.ndarray) -> pd.DataFrame:
    """Extract all model features from an RGB numpy array."""
    aspect_ratio = image_array.shape[1] / image_array.shape[0]

    features: dict = {}
    features.update(hsv(image_array))
    features.update(color_metadata(image_array))
    features.update(edges(image_array))
    features["aspect_ratio"] = aspect_ratio

    df = pd.DataFrame([features])
    df.drop(columns=COLS_TO_DROP, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def plot_edge_map(image_array: np.ndarray) -> plt.Figure:
    gray = rgb2gray(image_array)
    edge_map = sobel(gray)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Grayscale", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(edge_map, cmap="hot")
    axes[1].set_title("Edge Map (Sobel)", fontsize=12)
    axes[1].axis("off")
    fig.tight_layout()
    return fig


def plot_color_histograms(image_array: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    channels = [("Red", 0, "#e74c3c"), ("Green", 1, "#2ecc71"), ("Blue", 2, "#3498db")]
    for name, idx, color in channels:
        ax.hist(
            image_array[:, :, idx].flatten(),
            bins=64,
            alpha=0.55,
            color=color,
            label=name,
            density=True,
        )
    ax.set_xlabel("Pixel Intensity (0–255)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("RGB Channel Distribution", fontsize=13)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_shap_bars(shap_vals: np.ndarray, feature_names: list[str], label: str) -> plt.Figure:
    """Horizontal bar chart of SHAP values, coloured by contribution direction."""
    sorted_idx = np.argsort(np.abs(shap_vals))
    top_n = 15  # show top 15 features
    sorted_idx = sorted_idx[-top_n:]

    vals = shap_vals[sorted_idx]
    names = [feature_names[i] for i in sorted_idx]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(names, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"SHAP value  (← less {label}   more {label} →)", fontsize=10)
    ax.set_title(f"Feature contributions toward '{label}'", fontsize=12)
    for bar, val in zip(bars, vals):
        sign = "+" if val >= 0 else ""
        ax.text(
            val + (0.001 if val >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{val:.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
        )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI vs Human Image Classifier",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔍 AI vs Human Image Classifier")
st.markdown(
    "Upload an image and our Random Forest model will determine whether it was "
    "**AI-generated** or taken by a **real human**. The model achieves ~89% accuracy "
    "on a held-out test set."
)
st.divider()

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Drop an image here, or click to browse",
    type=["png", "jpg", "jpeg", "webp"],
    help="Supported formats: PNG, JPG, JPEG, WEBP",
)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

if uploaded_file is not None:
    # Load image
    image_bytes = uploaded_file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(pil_image)

    # Load model
    model = load_model()

    # Run feature extraction + prediction
    with st.spinner("Analyzing image..."):
        meta_df = extract_features(image_array)
        meta_df = meta_df[model.feature_names_in_]  # reorder to match training columns

        proba = model.predict_proba(meta_df)[0]
        ai_prob, human_prob = float(proba[0]), float(proba[1])
        is_human = human_prob > ai_prob
        label = "Human" if is_human else "AI Generated"
        confidence = max(ai_prob, human_prob)

    st.toast("Analysis complete!", icon="✅")

    # Celebration effects
    if is_human:
        st.balloons()
    else:
        st.snow()

    # -----------------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------------

    tab_result, tab_visual, tab_shap = st.tabs(
        ["Result", "Visual Analysis", "Feature Contributions"]
    )

    # --- Tab 1: Result ---
    with tab_result:
        col_img, col_result = st.columns([1, 1], gap="large")

        with col_img:
            st.image(pil_image, caption=f"Uploaded: {uploaded_file.name}", width="stretch")

        with col_result:
            if is_human:
                st.success(f"## Human Image")
                st.markdown(f"This image appears to be a **real photograph** taken by a human.")
            else:
                st.error(f"## AI Generated Image")
                st.markdown(f"This image appears to have been **generated by AI**.")

            st.markdown("---")

            # Confidence metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    label="Human probability",
                    value=f"{human_prob:.1%}",
                    delta=f"{human_prob - 0.5:+.1%} vs 50/50",
                    delta_color="normal",
                )
            with col_b:
                st.metric(
                    label="AI probability",
                    value=f"{ai_prob:.1%}",
                    delta=f"{ai_prob - 0.5:+.1%} vs 50/50",
                    delta_color="inverse",
                )

            st.markdown("**Confidence breakdown**")
            st.caption("Human")
            st.progress(human_prob)
            st.caption("AI Generated")
            st.progress(ai_prob)

            # Raw features expander
            with st.expander("Raw feature values"):
                display_df = meta_df.T.rename(columns={0: "Value"})
                display_df["Description"] = display_df.index.map(
                    lambda f: FEATURE_DESCRIPTIONS.get(f, "")
                )
                display_df["Value"] = display_df["Value"].map(lambda v: f"{v:.4f}")
                st.dataframe(display_df, width="stretch")

    # --- Tab 2: Visual Analysis ---
    with tab_visual:
        st.markdown(
            "These visualizations show the image properties that the model measured. "
            "The edge map reveals structural complexity; color distributions show channel balance."
        )

        vcol1, vcol2 = st.columns(2, gap="large")

        with vcol1:
            st.subheader("Edge Detection")
            st.caption(
                f"Edge entropy: **{meta_df['edge_entropy'].values[0]:.3f}** | "
                f"Edge density: **{meta_df['edge_density'].values[0]:.3f}**"
            )
            fig_edges = plot_edge_map(image_array)
            st.pyplot(fig_edges, width="stretch")

        with vcol2:
            st.subheader("Color Channels")
            st.caption(
                f"R mean: **{meta_df['r_mean'].values[0]:.1f}** | "
                f"G mean: **{meta_df['g_mean'].values[0]:.1f}** | "
                f"B mean: **{meta_df['b_mean'].values[0]:.1f}**"
            )
            fig_colors = plot_color_histograms(image_array)
            st.pyplot(fig_colors, width="stretch")

    # --- Tab 3: SHAP Feature Contributions ---
    with tab_shap:
        st.markdown(
            "**SHAP (SHapley Additive exPlanations)** shows exactly how much each feature "
            "contributed to this specific prediction. Green bars push toward the predicted label; "
            "red bars push against it."
        )

        with st.spinner("Computing feature contributions..."):
            explainer = get_explainer(model)
            shap_values = explainer.shap_values(meta_df)
            class_idx = 1 if is_human else 0
            # shap_values shape: (n_samples, n_features, n_classes) in SHAP >= 0.46
            # Index: sample 0, all features, target class
            shap_vals = shap_values[0, :, class_idx]

        fig_shap = plot_shap_bars(shap_vals, list(model.feature_names_in_), label)
        st.pyplot(fig_shap, width="stretch")

        st.info(
            "**How to read this chart:** Each bar shows how a feature's value for *this specific image* "
            "shifted the prediction. Longer bars = stronger influence. The sign tells you the direction."
        )

else:
    # Placeholder state
    st.info("Upload an image above to get started.")
    st.markdown(
        """
        **How it works:**
        1. The model extracts 23 hand-crafted features from your image — color statistics, edge properties, saturation, and aspect ratio
        2. A Random Forest classifier (300 trees, 92% test accuracy) predicts whether those features match AI-generated or human-captured images
        3. SHAP values reveal *which specific features* drove the prediction for your image
        """
    )
