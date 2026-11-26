"""Compare live-computed lexical features with the real values in manifests/url_test_manifest.csv.
Run from the project folder:  python verify_features.py
"""
import pandas as pd
from url_features import lexical_features

df = pd.read_csv("manifests/url_test_manifest.csv").sample(200, random_state=0)
cols = list(lexical_features("http://a.com").keys())
for c in cols:
    if c not in df.columns:
        print(f"{c:20s} NOT IN CSV (column name differs?)"); continue
    mine = df["url"].map(lambda u: lexical_features(str(u))[c])
    ok = ((mine - df[c]).abs() < 1e-3).mean()
    print(f"{c:20s} match {ok*100:5.1f}%" + ("" if ok > 0.95 else "   <-- check definition"))
