"""
Domain Matching Utilities
------------------------------
The screenshot dataset and the URL-features dataset are NOT row-aligned —
there's no shared ID between them. But screenshot filenames encode a
domain (e.g. "9_goldenfrog.com.png" -> "goldenfrog.com") and the URL
features file has full URLs (e.g. "https://www.goldenfrog.com/path").

This module extracts a comparable domain string from both sides so we can
join them and build an aligned image+URL subset for fusion training.
Rows that don't match anything (the majority, most likely — screenshots
and the 60K URL list weren't collected as pairs) are simply dropped; the
image and URL branches still get to see 100% of their own single-modality
data when trained/evaluated standalone.
"""

import re
from urllib.parse import urlparse


def domain_from_url(url: str) -> str:
    """https://www.goldenfrog.com/login?x=1 -> goldenfrog.com"""
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return ""
    netloc = netloc.split("@")[-1]  # strip any userinfo
    netloc = netloc.split(":")[0]  # strip port
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def domain_from_filename(filename: str) -> str:
    """9_goldenfrog.com.png -> goldenfrog.com
    Handles the "{index}_{domain}.png" naming pattern seen in the
    screenshot folders. Falls back gracefully if the pattern differs."""
    name = filename.rsplit(".", 1)[0]  # strip final extension (.png)
    # strip a leading "{digits}_" index prefix if present
    name = re.sub(r"^\d+_", "", name)
    return name.lower()


if __name__ == "__main__":
    assert domain_from_url("https://www.goldenfrog.com/login?x=1") == "goldenfrog.com"
    assert domain_from_url("http://sub.example.co.uk:8080/path") == "sub.example.co.uk"
    assert domain_from_filename("9_goldenfrog.com.png") == "goldenfrog.com"
    assert domain_from_filename("123_tailscale.com.png") == "tailscale.com"
    print("All domain-matching smoke tests passed.")
