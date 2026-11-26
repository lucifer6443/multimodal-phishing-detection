"""
Multimodal Spam/Phishing Detection - Demo Frontend
--------------------------------------------------------
Interactive Streamlit app to demo the four trained branches plus a combined verdict:
  - Text: paste any email text, get a live prediction
  - Image: upload a screenshot, get a live prediction
  - URL: pick a sample from the test set (features are pre-engineered,
    not recomputed live, since DNS/WHOIS lookups aren't part of this demo)
  - Fusion: pick a sample from the aligned image+URL set
  - Combined Verdict: text + screenshot + URL sample -> one final verdict
    (average of the branch scores, plus a 'Suspicious' flag if any one branch is >= 90%)

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
from url_features import extract_url_features
from screenshot import take_screenshot

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


def text_phish_prob(text):
    encoder, head = load_text_model()
    tokenized = encoder.tokenize([text])
    input_ids = tokenized["input_ids"].to(DEVICE)
    attention_mask = tokenized["attention_mask"].to(DEVICE)
    with torch.no_grad():
        logits = head(encoder(input_ids, attention_mask))
        return float(torch.softmax(logits, dim=1)[0, 1].cpu())


def image_phish_prob(image):
    encoder, head, transform = load_image_model()
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = head(encoder(tensor))
        return float(torch.softmax(logits, dim=1)[0, 1].cpu())


def url_phish_prob(row):
    encoder, head, scaler = load_url_model()
    features = row[FEATURE_COLUMNS].values.astype("float32").reshape(1, -1)
    tensor = torch.tensor(scaler.transform(features), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = head(encoder(tensor))
        return float(torch.softmax(logits, dim=1)[0, 1].cpu())


def url_phish_prob_from_url(url):
    """Live: compute the 18 features from a raw URL, then reuse the existing URL branch."""
    feats = extract_url_features(url)
    row = pd.Series(feats)
    missing = [c for c in FEATURE_COLUMNS if c not in row.index]
    if missing:
        raise KeyError(f"Feature names differ from FEATURE_COLUMNS: {missing}")
    return url_phish_prob(row)


def predict_bar(probs):
    """Renders a simple two-class probability bar chart."""
    st.write(f"**Legitimate:** {probs[0]:.1%}   |   **Phishing/Spam:** {probs[1]:.1%}")
    st.progress(float(probs[1]))


st.set_page_config(page_title="Phishing Detection Demo", layout="wide")
st.title("🛡️ Multimodal Spam/Phishing Detection — Live Demo")
st.caption("Preliminary model demo — Text, Image, URL, and Image+URL Fusion branches")

tab_checkall, tab_text, tab_image, tab_url, tab_fusion, tab_combined = st.tabs(
    ["✅ Check All", "📧 Text", "🖼️ Image", "🔗 URL", "🔗🖼️ Fusion (Image+URL)", "🗳️ Combined Verdict"]
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
        "⚠️ Note: only 4,046 image+URL pairs could be domain-matched between the two "
        "datasets, and only 21 of them are phishing. A possible reason is that phishing "
        "domains are short-lived and rarely overlap between separately collected datasets "
        "(this has not been tested). This branch is a proof-of-concept and does not predict "
        "phishing; it is reported honestly as a data limitation, not a bug."
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

# --- Combined Verdict tab (Text + Image + URL voting) ---
with tab_combined:
    st.subheader("Combined Verdict (Text + Image + URL)")
    st.write(
        "Give any of the three inputs. Each branch gives a phishing probability; "
        "the final verdict combines them. The Image+URL fusion branch is not used here."
    )
    st.caption(
        "Rule: average of the available branch scores >= 50% -> Phishing / Spam. "
        "Otherwise, if any single branch is >= 90% -> Suspicious. Otherwise -> Legitimate."
    )

    comb_text = st.text_area("Email text (optional)", value="", height=100, key="comb_text")
    comb_image_file = st.file_uploader(
        "Webpage screenshot (optional)", type=["png", "jpg", "jpeg"], key="comb_image"
    )

    use_url = st.checkbox("Include URL branch (pick a sample from the URL test set)", key="comb_use_url")
    comb_url_row = None
    if use_url:
        try:
            comb_url_df = pd.read_csv("manifests/url_test_manifest.csv")
            comb_idx = st.selectbox(
                "Choose a URL test sample",
                options=comb_url_df.index[:200],
                format_func=lambda i: comb_url_df.loc[i, "url"][:80],
                key="comb_url_idx",
            )
            comb_url_row = comb_url_df.loc[comb_idx]
            st.write("**Actual label:**", LABELS[int(comb_url_row["label_id"])])
        except FileNotFoundError:
            st.error("manifests/url_test_manifest.csv not found - run prepare_manifest.py first.")

    if st.button("Get Combined Verdict", key="comb_btn"):
        scores = {}
        if comb_text.strip():
            scores["Text"] = text_phish_prob(comb_text)
        if comb_image_file is not None:
            scores["Image"] = image_phish_prob(Image.open(comb_image_file).convert("RGB"))
        if comb_url_row is not None:
            scores["URL"] = url_phish_prob(comb_url_row)

        if not scores:
            st.warning("Please give at least one input (text, screenshot, or URL sample).")
        else:
            avg = sum(scores.values()) / len(scores)
            top_branch = max(scores, key=scores.get)
            top = scores[top_branch]

            if avg >= 0.5:
                st.error(f"Final verdict: **Phishing / Spam** (average score {avg:.1%})")
            elif top >= 0.9:
                st.warning(
                    f"Final verdict: **Suspicious** - average is {avg:.1%}, but the "
                    f"{top_branch} branch alone scored {top:.1%}"
                )
            else:
                st.success(f"Final verdict: **Legitimate** (average score {avg:.1%})")

            st.progress(float(avg))
            st.write("**Per-branch phishing probability:**")
            st.dataframe(
                pd.DataFrame(
                    {"Branch": list(scores.keys()), "Phishing probability": [f"{v:.1%}" for v in scores.values()]}
                ),
                hide_index=True,
            )


# --- Check All tab (one click: text + live URL + screenshot) ---
with tab_checkall:
    st.subheader("Check All: one click")
    st.write("Enter any of: message text, a URL, a screenshot. Only the inputs you give are used.")
    ca_text = st.text_area("Email / message text", height=100, key="ca_text")
    ca_url = st.text_input("URL (features are computed live)", key="ca_url")
    ca_auto = st.checkbox("Auto-capture a screenshot from the URL", value=True, key="ca_auto")
    ca_img = st.file_uploader("...or upload a screenshot (takes priority)",
                              type=["png", "jpg", "jpeg"], key="ca_img")

    if st.button("Check everything", type="primary", key="ca_btn"):
        scores, notes, shot = {}, [], None
        with st.spinner("Running all branches..."):
            if ca_text.strip():
                scores["Text"] = text_phish_prob(ca_text)
            if ca_url.strip():
                try:
                    scores["URL"] = url_phish_prob_from_url(ca_url.strip())
                except Exception as e:
                    notes.append(f"URL branch failed: {e}")
            if ca_img is not None:
                shot = Image.open(ca_img).convert("RGB")
            elif ca_url.strip() and ca_auto:
                try:
                    shot = Image.open(take_screenshot(ca_url.strip())).convert("RGB")
                except Exception as e:
                    notes.append(f"Screenshot failed (site blocked/offline or Playwright missing): {e}")
            if shot is not None:
                scores["Image"] = image_phish_prob(shot)

        for n in notes:
            st.warning(n)
        if not scores:
            st.warning("Please give at least one input.")
        else:
            avg = sum(scores.values()) / len(scores)
            top_branch = max(scores, key=scores.get)
            top = scores[top_branch]
            if avg >= 0.5:
                st.error(f"Final verdict: **Phishing / Spam** (average score {avg:.1%})")
            elif top >= 0.9:
                st.warning(f"Final verdict: **Suspicious** - average is {avg:.1%}, but the "
                           f"{top_branch} branch alone scored {top:.1%}")
            else:
                st.success(f"Final verdict: **Legitimate** (average score {avg:.1%})")
            st.progress(float(avg))
            cols = st.columns(len(scores))
            for col, (name, p) in zip(cols, scores.items()):
                col.metric(name, f"{p:.1%} phishing")
            if shot is not None:
                st.image(shot, caption="Screenshot used", width=320)
            st.caption(f"{len(scores)} of 3 branches ran.")