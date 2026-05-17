"""
Dual-Pathway ML Framework for Viral Hepatitis Classification
and Liver Fibrosis Assessment in an Ethiopian Clinical Setting.

Research demonstration app accompanying the MSc thesis by
Etsubdink Mersha Alem, Addis Ababa University, 2026.

NOT FOR CLINICAL USE.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent / "models"

# Module A artefacts (filenames match the notebook's save calls)
HBV_CALIBRATED_PATH = MODELS_DIR / "lr_hbv_calibrated.pkl"
HBV_RF_PATH = MODELS_DIR / "hbv_rf_for_shap.pkl"
HCV_RF_PATH = MODELS_DIR / "hcv_model_for_shap.pkl"
PREPROCESS_PATH = MODELS_DIR / "preprocessor.pkl"

# Module B artefacts
DENSENET_PATH = MODELS_DIR / "densenet121_final.pt"

CATEGORICAL_FEATURES = ["sex", "blood_group"]

BINARY_FEATURES = ["hiv", "vdrl", "tpha"]

# Numeric features expected by the preprocessor, in the exact order the
# notebook's NUMERIC_FEATURES list uses. The last three (ast_alt_ratio,
# apri, fib4) are derived from AST, ALT, age, and platelets and are
# computed on the fly in predict_tabular(); the user does not enter them.
NUMERIC_FEATURES_USER = [
    "age",
    "haemoglobin",
    "rbs",
    "creatinine",
    "ast",
    "alt",
    "platelets",
    "wbc",
    "rbc",
    "alp",
    "bun",
]
NUMERIC_FEATURES_DERIVED = ["ast_alt_ratio", "apri", "fib4"]
NUMERIC_FEATURES = NUMERIC_FEATURES_USER + NUMERIC_FEATURES_DERIVED

BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

METAVIR_STAGES = {
    0: "F0 (no fibrosis)",
    1: "F1 (portal fibrosis without septa)",
    2: "F2 (portal fibrosis with few septa)",
    3: "F3 (numerous septa without cirrhosis)",
    4: "F4 (cirrhosis)",
}

# ----------------------------------------------------------------------
# Demo-mode detection
# ----------------------------------------------------------------------


@dataclass
class ArtefactStatus:
    hbv_calibrated: bool
    hbv_rf: bool
    hcv_rf: bool
    preprocessor: bool
    densenet: bool

    @property
    def module_a_ready(self) -> bool:
        return self.hbv_calibrated and self.hbv_rf and self.hcv_rf and self.preprocessor

    @property
    def module_b_ready(self) -> bool:
        return self.densenet

    @property
    def any_demo_mode(self) -> bool:
        return not (self.module_a_ready and self.module_b_ready)


def check_artefacts() -> ArtefactStatus:
    return ArtefactStatus(
        hbv_calibrated=HBV_CALIBRATED_PATH.exists(),
        hbv_rf=HBV_RF_PATH.exists(),
        hcv_rf=HCV_RF_PATH.exists(),
        preprocessor=PREPROCESS_PATH.exists(),
        densenet=DENSENET_PATH.exists(),
    )


# ----------------------------------------------------------------------
# Module A loading and inference
# ----------------------------------------------------------------------


@st.cache_resource
def load_module_a():
    """Load tabular module artefacts.

    The notebook saves each artefact as a pickle of a wrapped dict, with
    the actual model under a known key (e.g., {'model': <estimator>,
    'feature_names_out': [...], 'note': '...'}). This loader extracts the
    estimator and the feature-name metadata from each wrapped dict.
    """
    import pickle

    # (display_name, dict_key_holding_model, file_path)
    artefact_specs = [
        ("preprocessor", "preprocessor", PREPROCESS_PATH),
        ("hbv_calibrated", "model", HBV_CALIBRATED_PATH),
        ("hbv_rf", "model", HBV_RF_PATH),
        ("hcv_rf", "model", HCV_RF_PATH),
    ]

    artefacts = {}
    feature_names_out = None

    for display_name, model_key, path in artefact_specs:
        if not path.exists():
            artefacts[display_name] = None
            continue
        try:
            with path.open("rb") as f:
                obj = pickle.load(f)
            # Wrapped dict format (from the notebook): extract under model_key
            if isinstance(obj, dict) and model_key in obj:
                artefacts[display_name] = obj[model_key]
                if feature_names_out is None and "feature_names_out" in obj:
                    feature_names_out = obj["feature_names_out"]
            else:
                # Raw model object format (fallback)
                artefacts[display_name] = obj
        except Exception as e:
            st.error(f"Failed to load {path.name}: {e}")
            artefacts[display_name] = None

    artefacts["feature_names_out"] = feature_names_out
    return artefacts


def _compute_derived_features(rec: dict) -> dict:
    """Compute the three derived numeric features from raw lab values."""
    ast = float(rec.get("ast", 0.0))
    alt = float(rec.get("alt", 0.0))
    age = float(rec.get("age", 0.0))
    plt = float(rec.get("platelets", 0.0))

    ast_alt_ratio = (ast / alt) if alt > 0 else 0.0
    apri = ((ast / 40.0) * (100.0 / plt) * 100.0) if plt > 0 else 0.0
    fib4 = ((age * ast) / (plt * (alt**0.5))) if (plt > 0 and alt > 0) else 0.0

    return {
        "ast_alt_ratio": ast_alt_ratio,
        "apri": apri,
        "fib4": fib4,
    }


def predict_tabular(input_record: dict, artefacts: dict) -> dict:
    """Run tabular inference. Returns dict with hbv_prob, hcv_prob,
    feature_vector (for SHAP), and shap_values."""
    import shap

    # Build a single-row DataFrame matching the training schema exactly.
    # The preprocessor expects 19 columns: 14 numeric + 3 binary + 2 cat.
    full_record = dict(input_record)
    full_record.update(_compute_derived_features(input_record))

    expected_cols = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    row = pd.DataFrame([{col: full_record.get(col) for col in expected_cols}])

    # Preprocess
    X = artefacts["preprocessor"].transform(row)
    # ColumnTransformer may return a DataFrame depending on set_output config
    X_arr = X.values if hasattr(X, "values") else X

    # Calibrated HBV probability
    hbv_prob = float(artefacts["hbv_calibrated"].predict_proba(X_arr)[0, 1])

    # HCV probability from the grid winner (RF in the thesis); reported with
    # a low-confidence caveat in the UI because the thesis frames the HCV
    # result as a data-sufficiency null finding.
    hcv_prob = float(artefacts["hcv_rf"].predict_proba(X_arr)[0, 1])

    # SHAP attribution on the HBV random forest (Tree SHAP exact).
    explainer = shap.TreeExplainer(artefacts["hbv_rf"])
    shap_values = explainer.shap_values(X_arr)

    # shap_values may be a list [neg, pos] or a 3D array (n,F,2);
    # extract the positive-class slice for our single sample.
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
        base = explainer.expected_value[1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        shap_vals = shap_values[0, :, 1]
        base = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
    else:
        shap_vals = shap_values[0]
        base = explainer.expected_value

    # Feature names: prefer the captured feature_names_out, else fall back
    feature_names = artefacts.get("feature_names_out") or list(range(len(shap_vals)))

    return {
        "hbv_prob": hbv_prob,
        "hcv_prob": hcv_prob,
        "feature_names": list(feature_names),
        "feature_values": X_arr[0].tolist(),
        "shap_values": shap_vals.tolist()
        if hasattr(shap_vals, "tolist")
        else list(shap_vals),
        "shap_base_value": float(base),
    }


def predict_tabular_demo(input_record: dict) -> dict:
    """Stubbed predictions used when models are absent."""
    # Heuristic: scale ALT and AST relative to typical normal range
    alt = input_record.get("alt", 30.0)
    ast = input_record.get("ast", 30.0)
    plt = input_record.get("platelets", 250.0)
    fake_hbv = min(0.5, max(0.01, (alt + ast - 60.0) / 200.0 + (250.0 - plt) / 1000.0))
    fake_hcv = min(0.3, max(0.005, fake_hbv * 0.3))
    fake_features = NUMERIC_FEATURES + ["apri", "fib4", "sex_M"]
    fake_shap = np.random.RandomState(42).normal(0, 0.02, len(fake_features)).tolist()
    return {
        "hbv_prob": float(fake_hbv),
        "hcv_prob": float(fake_hcv),
        "feature_names": fake_features,
        "feature_values": [input_record.get(f, 0.0) for f in NUMERIC_FEATURES]
        + [0.0, 0.0, 1.0],
        "shap_values": fake_shap,
        "shap_base_value": 0.012,
    }


# ----------------------------------------------------------------------
# Module B loading and inference
# ----------------------------------------------------------------------


@st.cache_resource
def load_module_b():
    """Load DenseNet-121 fibrosis classifier."""
    if not DENSENET_PATH.exists():
        return None
    try:
        import torch
        from torchvision import models

        model = models.densenet121(weights=None)
        # Replace classifier head: 5 METAVIR classes
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, 5)
        state = torch.load(DENSENET_PATH, map_location="cpu")
        # Handle both raw state_dict and wrapped dict
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load DenseNet-121: {e}")
        return None


def predict_ultrasound(image_bytes: bytes, model) -> dict:
    """Run ultrasound inference and produce Grad-CAM overlay."""
    import torch
    from PIL import Image
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from torchvision import transforms

    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_np = np.array(pil.resize((224, 224))) / 255.0

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    tensor = preprocess(pil).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

    predicted_idx = int(np.argmax(probs))

    # Grad-CAM at the final convolutional block of DenseNet-121
    target_layer = model.features.norm5
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=tensor)[0]
    overlay = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)

    return {
        "probabilities": probs,
        "predicted_index": predicted_idx,
        "predicted_label": METAVIR_STAGES[predicted_idx],
        "gradcam_overlay": overlay,
        "original_image": original_np,
    }


def predict_ultrasound_demo(image_bytes: bytes) -> dict:
    """Stubbed prediction returning the input image with no real overlay."""
    from PIL import Image

    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(pil.resize((224, 224))) / 255.0
    # Fake probabilities biased toward F0/F4 to mimic the corpus bias
    fake_probs = [0.45, 0.10, 0.08, 0.12, 0.25]
    return {
        "probabilities": fake_probs,
        "predicted_index": 0,
        "predicted_label": METAVIR_STAGES[0],
        "gradcam_overlay": (arr * 255).astype(np.uint8),
        "original_image": arr,
    }


# ----------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------

SAMPLE_TABULAR_RECORD = {
    # demographics
    "age": 45,
    "sex": "M",
    "blood_group": "O+",
    # numeric labs
    "haemoglobin": 13.2,
    "rbs": 98,  # random blood sugar
    "creatinine": 0.9,
    "ast": 65,  # elevated
    "alt": 78,  # elevated
    "platelets": 142,  # low-normal
    "wbc": 6.8,
    "rbc": 4.7,
    "alp": 110,
    "bun": 15,
    # binary serology (0 = negative, 1 = positive)
    "hiv": 0,
    "vdrl": 0,
    "tpha": 0,
}


def caveat_banner(status: ArtefactStatus) -> None:
    if status.any_demo_mode:
        missing = []
        if not status.module_a_ready:
            missing.append("Module A (tabular)")
        if not status.module_b_ready:
            missing.append("Module B (ultrasound)")
        st.warning(
            f"**Demo mode active for: {', '.join(missing)}.** "
            "Predictions shown are stubbed and not from the trained models. "
            f"Place the saved artefacts in `{MODELS_DIR}/` to enable real inference."
        )

    st.info(
        "**Research demonstration only — not for clinical use.** "
        "This system is the artefact accompanying an MSc thesis. "
        "Inputs are not stored. No clinical decisions should be based on the outputs."
    )


def render_tabular_form() -> dict | None:
    st.subheader("Tabular pathway — routine laboratory inputs")
    st.caption(
        "Fill in the patient's demographics, routine laboratory values, and "
        "auxiliary serology results, or use the sample-record button. The "
        "three derived features (AST/ALT ratio, APRI, FIB-4) are computed "
        "automatically from the entered values."
    )

    use_sample = st.button("Load sample patient record")

    # Specification: each tuple is (key, label, min, max, default).
    # Only user-entered numeric features appear here; ast_alt_ratio, apri, fib4
    # are derived inside predict_tabular() and are not part of the form.
    numeric_inputs = [
        ("haemoglobin", "Haemoglobin (g/dL)", 5.0, 20.0, 14.0),
        ("rbs", "Random blood sugar (mg/dL)", 40.0, 500.0, 95.0),
        ("creatinine", "Creatinine (mg/dL)", 0.2, 15.0, 0.9),
        ("ast", "AST (U/L)", 0.0, 1000.0, 30.0),
        ("alt", "ALT (U/L)", 0.0, 1000.0, 30.0),
        ("platelets", "Platelets (×10⁹/L)", 10.0, 800.0, 250.0),
        ("wbc", "WBC (×10⁹/L)", 0.5, 30.0, 7.0),
        ("rbc", "RBC (×10¹²/L)", 2.0, 7.0, 4.8),
        ("alp", "ALP (U/L)", 0.0, 1000.0, 80.0),
        ("bun", "BUN (mg/dL)", 2.0, 200.0, 14.0),
    ]
    field_defaults: dict = {key: default for key, _, _, _, default in numeric_inputs}
    field_defaults.update(
        {
            "age": 40,
            "sex": "M",
            "blood_group": "O+",
            "hiv": 0,
            "vdrl": 0,
            "tpha": 0,
        }
    )

    if use_sample:
        st.session_state["tabular_record"] = SAMPLE_TABULAR_RECORD.copy()

    defaults = st.session_state.get("tabular_record", field_defaults)

    record: dict = {}

    # --- Demographics row ---
    st.markdown("**Demographics**")
    dcols = st.columns(3)
    record["age"] = dcols[0].number_input(
        "Age (years)",
        min_value=15,
        max_value=100,
        value=int(defaults.get("age", 40)),
        step=1,
    )
    record["sex"] = dcols[1].selectbox(
        "Sex",
        options=["M", "F"],
        index=0 if defaults.get("sex", "M") == "M" else 1,
    )
    bg_default = defaults.get("blood_group", "O+")
    bg_index = (
        BLOOD_GROUPS.index(bg_default)
        if bg_default in BLOOD_GROUPS
        else BLOOD_GROUPS.index("O+")
    )
    record["blood_group"] = dcols[2].selectbox(
        "Blood group",
        options=BLOOD_GROUPS,
        index=bg_index,
    )

    # --- Numeric labs ---
    st.markdown("**Routine laboratory values**")
    ncols = st.columns(3)
    for idx, (key, label, mn, mx, default) in enumerate(numeric_inputs):
        col = ncols[idx % 3]
        raw = float(defaults.get(key, default))
        clamped = min(max(raw, mn), mx)
        record[key] = col.number_input(
            label,
            min_value=mn,
            max_value=mx,
            value=clamped,
            step=0.1,
            key=f"num_{key}",
        )

    # --- Binary serology row ---
    st.markdown("**Auxiliary serology (negative / positive)**")
    bcols = st.columns(3)
    for i, (key, label) in enumerate(
        [
            ("hiv", "HIV"),
            ("vdrl", "VDRL"),
            ("tpha", "TPHA"),
        ]
    ):
        opt = bcols[i].selectbox(
            label,
            options=["Negative", "Positive"],
            index=0 if int(defaults.get(key, 0)) == 0 else 1,
            key=f"bin_{key}",
        )
        record[key] = 1 if opt == "Positive" else 0

    st.markdown("---")

    if st.button("Run tabular pathway", type="primary"):
        return record
    return None


def render_tabular_results(result: dict) -> None:
    st.subheader("Tabular pathway — predicted serological status")

    col_hbv, col_hcv = st.columns(2)
    col_hbv.metric(
        "Hepatitis B probability",
        f"{result['hbv_prob']:.1%}",
        help="Calibrated logistic regression output. Brier-validated probability scale.",
    )
    col_hcv.metric(
        "Hepatitis C probability",
        f"{result['hcv_prob']:.1%}",
        help="Random-forest output. Reported as data-sufficiency null finding in the thesis; treat as low-confidence.",
    )

    st.caption(
        "**Caveat (HCV):** the underlying cohort had only 33 HCV-positive patients "
        "(prevalence 0.073 percent). The HCV classifier is reported in the thesis as a "
        "null finding; predictions shown here are not validated above the no-skill threshold "
        "and should not be acted on."
    )

    st.divider()
    st.subheader("Feature attribution (SHAP) — top contributors")

    shap_df = pd.DataFrame(
        {
            "feature": result["feature_names"],
            "value": result["feature_values"],
            "shap": result["shap_values"],
        }
    )
    shap_df["abs_shap"] = shap_df["shap"].abs()
    top = shap_df.sort_values("abs_shap", ascending=False).head(8)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top["shap"]]
    ax.barh(top["feature"], top["shap"], color=colors)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("SHAP contribution (positive → higher HBV probability)")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)


def render_ultrasound_input() -> bytes | None:
    st.subheader("Imaging pathway — liver ultrasound input")
    upload = st.file_uploader(
        "Upload a B-mode liver ultrasound image (JPEG or PNG)",
        type=["jpg", "jpeg", "png"],
    )
    if upload is not None:
        if st.button("Run imaging pathway", type="primary"):
            return upload.getvalue()
    return None


def render_ultrasound_results(result: dict) -> None:
    st.subheader("Imaging pathway — predicted fibrosis stage")

    st.metric("Predicted METAVIR stage", result["predicted_label"])

    probs_df = pd.DataFrame(
        {
            "stage": [METAVIR_STAGES[i] for i in range(5)],
            "probability": result["probabilities"],
        }
    )
    st.bar_chart(probs_df.set_index("stage"))

    st.caption(
        "**Caveat:** the underlying public dataset does not preserve patient "
        "identifiers, so the within-corpus accuracy reported in the thesis "
        "(macro-F1 0.94, accuracy 0.96) is an upper bound on patient-disjoint "
        "performance. Treat outputs as within-dataset demonstrations."
    )

    st.divider()
    st.subheader("Visual attribution (Grad-CAM)")

    cols = st.columns(2)
    cols[0].image(
        result["original_image"],
        caption="Input image (resized 224×224)",
        use_container_width=True,
    )
    cols[1].image(
        result["gradcam_overlay"],
        caption="Grad-CAM overlay at final conv block",
        use_container_width=True,
    )


# ----------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="Dual-Pathway Hepatitis & Fibrosis Demo",
        page_icon="🧪",
        layout="wide",
    )

    st.title("Dual-Pathway ML Framework: Demonstration")
    st.markdown(
        "Tabular hepatitis classifier (Module A) and liver-ultrasound fibrosis "
        "classifier (Module B) from the thesis *A Dual-Pathway Machine Learning "
        "Framework for Viral Hepatitis Classification and Liver Fibrosis "
        "Assessment in an Ethiopian Clinical Setting* (Etsubdink Mersha Alem, "
        "Addis Ababa University, 2026)."
    )

    status = check_artefacts()
    caveat_banner(status)

    tab_a, tab_b, tab_about = st.tabs(
        ["Module A (Tabular)", "Module B (Ultrasound)", "About"]
    )

    with tab_a:
        record = render_tabular_form()
        if record is not None:
            if status.module_a_ready:
                artefacts = load_module_a()
                with st.spinner("Running tabular inference..."):
                    result = predict_tabular(record, artefacts)
            else:
                with st.spinner("Demo mode — generating stubbed prediction..."):
                    result = predict_tabular_demo(record)
            render_tabular_results(result)

    with tab_b:
        image_bytes = render_ultrasound_input()
        if image_bytes is not None:
            if status.module_b_ready:
                model = load_module_b()
                if model is not None:
                    with st.spinner("Running DenseNet-121 inference and Grad-CAM..."):
                        result = predict_ultrasound(image_bytes, model)
                    render_ultrasound_results(result)
                else:
                    st.error(
                        "DenseNet-121 weights present but failed to load. See error above."
                    )
            else:
                with st.spinner("Demo mode — returning stubbed prediction..."):
                    result = predict_ultrasound_demo(image_bytes)
                render_ultrasound_results(result)

    with tab_about:
        st.markdown("""
### About this demonstration

This streamlit application is the deployable instantiation of the dual-pathway
framework described in Chapter 4 of the accompanying thesis. The framework
integrates two independently trained modules:

- **Module A** maps routine laboratory profiles to predicted hepatitis B and
  hepatitis C serological status, with SHAP-based feature attribution.
  Trained on 45,015 ambulatory-laboratory records from a private diagnostic
  facility in Addis Ababa.

- **Module B** maps B-mode liver ultrasound images to one of five METAVIR
  fibrosis stages, with Grad-CAM visual attribution. Trained on a public
  corpus of 6,323 images.

The two modules are integrated by *composition* rather than by joint training,
which permits each module to operate on data the other module's training set
does not contain.

### Honest reporting

This application surfaces the same caveats stated in the thesis:

1. The Module A hepatitis B classifier achieved modest test discrimination
   (AUC 0.73) on the Ethiopian cohort; its calibrated probabilities are
   suitable for triage support but not diagnostic confirmation.

2. The Module A hepatitis C classifier is a documented **null finding** tied
   to the cohort's 0.073% HCV prevalence. Outputs are shown for completeness
   but should not be acted on.

3. The Module B fibrosis classifier achieves high within-corpus accuracy
   (macro-F1 0.94, accuracy 0.96), but the public dataset does not preserve
   patient identifiers; the reported numbers are an upper bound on
   patient-disjoint performance.

### Not for clinical use

This is a research artefact accompanying an MSc thesis. It is not a
medical device, has not undergone regulatory evaluation, and must not be
used to inform clinical decisions about real patients.

### Author and supervisor

- **Author:** Etsubdink Mersha Alem
- **Institution:** Addis Ababa University, Department of Computer Science
- **Year:** 2026

### Repository

Source code and trained weights are available at:
[https://github.com/etsubdink/hepatitis-fibrosis-framework](https://github.com/etsubdink/hepatitis-fibrosis-framework)
*(update this URL to your actual repository)*
""")


if __name__ == "__main__":
    main()
