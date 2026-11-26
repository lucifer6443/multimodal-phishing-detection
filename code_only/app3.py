"""
Multimodal Spam/Phishing Detection - Demo Frontend
--------------------------------------------------------
Interactive Streamlit app to demo all four trained branches:
  - Text: paste any email text, get a live prediction
  - Image: upload a screenshot, get a live prediction
  - URL: pick a sample from the test set (features are pre-engineered,
    not recomputed live, since DNS/WHOIS lookups aren't part of this demo)
  - Fusion: pick a sample from the aligned image+URL set

Run with:
    streamlit run app.py

Requires all four checkpoints to already exist in ./checkpoints/ and the
manifests/scaler to exist in ./manifests/ (i.e. run after training).
"""

import pickle

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

from models.text_encoder import TextEncoder, TextClassifierHead
from models.image_encoder import ImageEncoder, ImageClassifierHead, get_recommended_transform
from models.url_feature_encoder import URLFeatureEncoder, URLFeatureClassifierHead
from models.fusion_model import ImageURLFusionDetector
from data.feature_columns import FEATURE_COLUMNS

LABELS = {0: "Legitimate", 1: "Phishing / Spam"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_text_model():
    encoder = TextEncoder().to(DEVICE)
    head = TextClassifierHead().to(DEVICE)
    combined = nn.ModuleDict({"text_encoder": encoder, "text_head": head})
    combined.load_state_dict(torch.load("checkpoints/text_model_best.pt", map_location=DEVICE))
    encoder.eval()
    head.eval()
    return encoder, head


@st.cache_resource
def load_image_model():
    encoder = ImageEncoder(backbone="resnet50").to(DEVICE)
    head = ImageClassifierHead().to(DEVICE)
    combined = nn.ModuleDict({"image_encoder": encoder, "head": head})
    combined.load_state_dict(torch.load("checkpoints/image_model_best.pt", map_location=DEVICE))
    encoder.eval()
    head.eval()
    transform = get_recommended_transform(backbone="resnet50")
    return encoder, head, transform


@st.cache_resource
def load_url_model():
    encoder = URLFeatureEncoder(num_features=18).to(DEVICE)
    head = URLFeatureClassifierHead().to(DEVICE)
    combined = nn.ModuleDict({"url_encoder": encoder, "head": head})
    combined.load_state_dict(torch.load("checkpoints/url_model_best.pt", map_location=DEVICE))
    encoder.eval()
    head.eval()
    with open("manifests/url_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return encoder, head, scaler


@st.cache_resource
def load_fusion_model():
    model = ImageURLFusionDetector(image_backbone="resnet50").to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/fusion_model_best.pt", map_location=DEVICE))
    model.eval()
    transform = get_recommended_transform(backbone="resnet50")
    with open("manifests/url_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, transform, scaler


def predict_bar(probs):
    """Renders a simple two-class probability bar chart."""
    st.write(f"**Legitimate:** {probs[0]:.1%}   |   **Phishing/Spam:** {probs[1]:.1%}")
    st.progress(float(probs[1]))


st.set_page_config(page_title="Phishing Detection Demo", layout="wide")
st.title("🛡️ Multimodal Spam/Phishing Detection — Live Demo")
st.caption("Preliminary model demo — Text, Image, URL, and Image+URL Fusion branches")

tab_text, tab_image, tab_url, tab_fusion = st.tabs(
    ["📧 Text", "🖼️ Image", "🔗 URL", "🔗🖼️ Fusion (Image+URL)"]
)

# --- Text tab ---
with tab_text:
    st.subheader("Email Text Classifier")
    st.write("Paste any email subject + body text below.")
    default_text = "Congratulations! You have won a free prize. Click here to claim now before it expires!"
    user_text = st.text_area("Email text", value=default_text, height=120)

    if st.button("Classify Email", key="text_btn"):
        encoder, head = load_text_model()
        tokenized = encoder.tokenize([user_text])
        input_ids = tokenized["input_ids"].to(DEVICE)
        attention_mask = tokenized["attention_mask"].to(DEVICE)

        with torch.no_grad():
            embedding = encoder(input_ids, attention_mask)
            logits = head(embedding)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(probs.argmax())

        st.success(f"Prediction: **{LABELS[pred]}**")
        predict_bar(probs)

# --- Image tab ---
with tab_image:
    st.subheader("Webpage Screenshot Classifier")
    uploaded_image = st.file_uploader("Upload a webpage screenshot", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded screenshot", use_container_width=True)

        if st.button("Classify Screenshot", key="image_btn"):
            encoder, head, transform = load_image_model()
            tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                embedding = encoder(tensor)
                logits = head(embedding)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred = int(probs.argmax())

            with col2:
                st.success(f"Prediction: **{LABELS[pred]}**")
                predict_bar(probs)

# --- URL tab ---
with tab_url:
    st.subheader("URL Feature Classifier")
    st.write("Pick a sample from the held-out URL test set (60K-URL dataset).")

    try:
        url_test_df = pd.read_csv("manifests/url_test_manifest.csv")
        sample_idx = st.selectbox(
            "Choose a test sample",
            options=url_test_df.index[:200],  # cap dropdown length for usability
            format_func=lambda i: url_test_df.loc[i, "url"][:80],
        )
        row = url_test_df.loc[sample_idx]
        st.write("**URL:**", row["url"])
        st.write("**Actual label:**", LABELS[int(row["label_id"])])

        with st.expander("View 18 engineered features"):
            st.dataframe(row[FEATURE_COLUMNS].to_frame(name="value"))

        if st.button("Classify URL", key="url_btn"):
            encoder, head, scaler = load_url_model()
            features = row[FEATURE_COLUMNS].values.astype("float32").reshape(1, -1)
            features_scaled = scaler.transform(features)
            tensor = torch.tensor(features_scaled, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                embedding = encoder(tensor)
                logits = head(embedding)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred = int(probs.argmax())

            st.success(f"Prediction: **{LABELS[pred]}**")
            predict_bar(probs)
    except FileNotFoundError:
        st.error("manifests/url_test_manifest.csv not found — run prepare_manifest.py first.")

# --- Fusion tab ---
with tab_fusion:
    st.subheader("Image + URL Fusion Classifier")
    st.caption(
        "⚠️ Note: only ~4,046 image+URL pairs could be domain-matched between the two "
        "datasets, and phishing sites were rare among the matches (short-lived domains "
        "rarely overlap between separately-collected datasets). This branch is a "
        "proof-of-concept and currently underperforms on phishing recall — reported "
        "honestly as a data limitation, not a bug."
    )

    try:
        aligned_df = pd.read_csv("manifests/aligned_image_url_manifest.csv")
        sample_idx = st.selectbox(
            "Choose an aligned sample",
            options=aligned_df.index[:200],
            format_func=lambda i: aligned_df.loc[i, "domain"],
        )
        row = aligned_df.loc[sample_idx]
        st.write("**Domain:**", row["domain"])
        st.write("**Actual label:**", LABELS[int(row["label_id"])])

        try:
            image = Image.open(row["image_path"]).convert("RGB")
            st.image(image, caption="Screenshot", width=300)
        except Exception:
            st.warning("Could not load screenshot file for this row.")
            image = None

        if st.button("Classify (Image + URL)", key="fusion_btn") and image is not None:
            model, transform, scaler = load_fusion_model()
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            features = row[FEATURE_COLUMNS].values.astype("float32").reshape(1, -1)
            features_scaled = scaler.transform(features)
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                logits = model(image_tensor, features_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred = int(probs.argmax())

            st.success(f"Prediction: **{LABELS[pred]}**")
            predict_bar(probs)
    except FileNotFoundError:
        st.error("manifests/aligned_image_url_manifest.csv not found — run prepare_manifest.py first.")
