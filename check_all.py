"""One-click 'Check All' tab for the Streamlit demo.

Usage in app.py:
    from check_all import render_check_all_tab
    tab_all, tab_text, tab_img, tab_url = st.tabs(["Check All", "Text", "Image", "URL"])
    with tab_all:
        render_check_all_tab(predict_text, predict_image, predict_url_features)

You pass in three functions from your existing code, each returning P(phishing) in 0..1:
    predict_text(text: str) -> float
    predict_image(image) -> float         # receives a PIL.Image
    predict_url_features(feats: dict) -> float   # dict from extract_url_features();
        scale it with your fitted scaler (fit_url_scaler.py) in the SAME column order, then run the MLP.
"""
import streamlit as st
from PIL import Image
from url_features import extract_url_features


def combine(scores: dict):
    """Demo rule from the paper: mean >= 50% -> Phishing; else any >= 90% -> Suspicious."""
    vals = list(scores.values())
    if sum(vals) / len(vals) >= 0.5:
        return "Phishing / Spam", "error"
    if any(v >= 0.9 for v in vals):
        return "Suspicious", "warning"
    return "Legitimate", "success"


def render_check_all_tab(predict_text, predict_image, predict_url_features):
    st.subheader("One-click check")
    text = st.text_area("Email / message text", key="all_text")
    url = st.text_input("URL", key="all_url")
    auto_shot = st.checkbox("Auto-capture screenshot from the URL", value=True, key="all_auto")
    uploaded = st.file_uploader("...or upload a screenshot", type=["png", "jpg", "jpeg"], key="all_img")

    if not st.button("Check everything", type="primary", key="all_go"):
        return

    scores, notes, shot = {}, [], None
    with st.spinner("Running all branches..."):
        if text.strip():
            scores["Text"] = float(predict_text(text))
        if url.strip():
            try:
                feats = extract_url_features(url.strip())
                scores["URL"] = float(predict_url_features(feats))
            except Exception as e:
                notes.append(f"URL branch failed: {e}")
        if uploaded is not None:
            shot = Image.open(uploaded).convert("RGB")
        elif url.strip() and auto_shot:
            try:
                from screenshot import take_screenshot
                shot = Image.open(take_screenshot(url.strip())).convert("RGB")
            except Exception as e:
                notes.append(f"Screenshot failed (site blocked/offline or Playwright missing): {e}")
        if shot is not None:
            scores["Image"] = float(predict_image(shot))

    for n in notes:
        st.warning(n)
    if not scores:
        st.warning("Give at least one input (text, URL or screenshot).")
        return
    verdict, level = combine(scores)
    getattr(st, level)(f"Verdict: {verdict}")
    cols = st.columns(len(scores))
    for col, (name, p) in zip(cols, scores.items()):
        col.metric(name, f"{p*100:.1f}% phishing")
        col.progress(min(max(p, 0.0), 1.0))
    if shot is not None:
        st.image(shot, caption="Screenshot used", width=320)
    if len(scores) < 3:
        st.caption(f"Only {len(scores)} of 3 branches ran; the verdict uses the available scores.")
