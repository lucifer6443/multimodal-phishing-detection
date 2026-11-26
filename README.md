# Multimodal Spam / Phishing Detection

A preliminary multi-branch system that detects spam and phishing from three kinds of evidence: **email text**, **webpage screenshots** and **URL features**. Includes a Streamlit demo with a one-click **Check All** tab.

> Prototype for a final-year B.Tech project at Galgotias University. It is not a validated production detector.

## Branches
| Branch | Model | Input |
|---|---|---|
| Text | Fine-tuned DistilBERT | Email subject + body |
| Image | Fine-tuned ResNet50 | Webpage screenshot (224x224) |
| URL | MLP (18 → 128 → 256 → 256) | 18 engineered URL features |
| Image + URL fusion | Late fusion of image and URL embeddings | Matched screenshot + URL (proof of concept, not used in the verdict) |

The three data sources were collected independently and share no identifier, so the branches are trained separately. The demo combines their scores with a simple rule: average ≥ 50% → Phishing/Spam; otherwise any single score ≥ 90% → Suspicious; otherwise Legitimate.

## Check All tab
Enter any of: message text, a URL, a screenshot. For a URL, the 18 features are computed live (lexical features plus DNS/WHOIS lookups) and a screenshot can be captured automatically with Playwright. Only the inputs you give are used.

## Run it
```bash
pip install -r requirements.txt
playwright install chromium
streamlit run app1.py
```
Datasets and trained checkpoints are **not included** in this repository. Training scripts: `train_text.py`, `train_image.py`, `train_url.py`, `train_fusion.py`. Run `python verify_features.py` to compare the live URL features with the dataset values.

## Data
- Website URL features and screenshots: *Phishing and Legitimate Websites Dataset* v1.0 (Zenodo, CC BY 4.0).
- Email text: [sniff07/phishing-dataset](https://github.com/sniff07/phishing-dataset). The repository states no licence, so please download it from the source.

## Limitations
CPU-only training with reduced epochs and a single random seed. The image+URL fusion model did not detect phishing because only 21 of 4,046 matched pairs are phishing. Live URL features may differ slightly from the dataset's own feature extraction.

## Authors
Aviral Singh, Aarish Ansari, Dr. Mohd Zuhair (Galgotias University).
