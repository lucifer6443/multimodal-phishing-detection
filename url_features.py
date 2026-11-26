"""Live extraction of the 18 URL features used by the URL branch.

Column order must match merged_features_final.xlsx (minus 'url' and 'label'):
entropy, url_length, num_www, ratio_digits_url, num_subdomains, length_hostname,
random_domain, num_dots, num_hyphens, ratio_digits_host, prefix_suffix,
num_question_mark, num_eq, domain_age, dns_resolvable, num_ips, ttl, num_name_servers

IMPORTANT: I could not see how the dataset authors defined each feature, so the
definitions below are best guesses. Run `python url_features.py <xlsx>` to compare
them with the real values in your Excel file and fix any column that mismatches.
"""
import math
import re
import socket
import sys
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlparse

FEATURE_COLUMNS = [
    "entropy", "url_length", "num_www", "ratio_digits_url", "num_subdomains",
    "length_hostname", "random_domain", "num_dots", "num_hyphens",
    "ratio_digits_host", "prefix_suffix", "num_question_mark", "num_eq",
    "domain_age", "dns_resolvable", "num_ips", "ttl", "num_name_servers",
]
NETWORK_DEFAULTS = {"domain_age": 0, "dns_resolvable": 0, "num_ips": 0, "ttl": 0, "num_name_servers": 0}


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _split_host(host: str):
    parts = host.split(".")
    try:
        import tldextract  # optional, more accurate
        ext = tldextract.extract(host)
        return ext.subdomain, ext.domain, ext.suffix
    except Exception:
        if len(parts) <= 2:
            return "", parts[0] if parts else "", parts[-1] if len(parts) > 1 else ""
        return ".".join(parts[:-2]), parts[-2], parts[-1]


def _random_domain(domain: str) -> int:
    """Dataset rule inferred from examples: 1 if the domain label has no vowels."""
    return int(not any(ch in "aeiou" for ch in domain.lower()))

def lexical_features(url: str) -> dict:
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    p = urlparse(url)
    host = (p.hostname or "").lower()
    sub, domain, _ = _split_host(host)
    digits_url = sum(c.isdigit() for c in url)
    digits_host = sum(c.isdigit() for c in host)
    return {
        "entropy": _entropy(url),
        "url_length": len(url),
        "num_www": url.lower().count("www"),
        "ratio_digits_url": digits_url / len(url) if url else 0,
        "num_subdomains": len([s for s in sub.split(".") if s]),
        "length_hostname": len(host),
        "random_domain": _random_domain(domain),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "ratio_digits_host": digits_host / len(host) if host else 0,
        "prefix_suffix": int("-" in domain),
        "num_question_mark": url.count("?"),
        "num_eq": url.count("="),
    }


def network_features(url: str, timeout: float = 4.0) -> dict:
    """domain_age (days), dns_resolvable, num_ips, ttl, num_name_servers.
    Needs `dnspython` and `python-whois`; falls back to 0 if a lookup fails."""
    out = dict(NETWORK_DEFAULTS)
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return out
    try:
        import dns.resolver
        res = dns.resolver.Resolver()
        res.lifetime = timeout
        ans = res.resolve(host, "A")
        out["dns_resolvable"] = 1
        out["num_ips"] = len(ans)
        out["ttl"] = int(ans.rrset.ttl)
    except Exception:
        try:
            ips = {ai[4][0] for ai in socket.getaddrinfo(host, None)}
            out["dns_resolvable"], out["num_ips"] = 1, len(ips)
        except Exception:
            pass
    try:
        import dns.resolver
        _, domain, suffix = _split_host(host)
        reg = f"{domain}.{suffix}" if suffix else host
        out["num_name_servers"] = len(dns.resolver.resolve(reg, "NS", lifetime=timeout))
    except Exception:
        pass
    try:
        import whois
        w = whois.whois(host)
        created = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
        if created:
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            out["domain_age"] = max((datetime.now(timezone.utc) - created).days, 0)
    except Exception:
        pass
    return out


def extract_url_features(url: str, live_network: bool = True) -> dict:
    feats = lexical_features(url)
    feats.update(network_features(url) if live_network else NETWORK_DEFAULTS)
    return {k: feats[k] for k in FEATURE_COLUMNS}


def verify_against_dataset(xlsx_path: str, n: int = 50):
    """Compare lexical features computed here with the Excel values."""
    import pandas as pd
    df = pd.read_excel(xlsx_path).sample(n, random_state=0)
    lex = [c for c in FEATURE_COLUMNS if c in lexical_features("http://a.com")]
    for col in lex:
        mine = df["url"].map(lambda u: lexical_features(str(u))[col])
        ok = (abs(mine - df[col]) < 1e-3).mean()
        print(f"{col:20s} match {ok*100:5.1f}%" + ("" if ok > 0.95 else "   <-- fix definition"))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        verify_against_dataset(sys.argv[1])
    else:
        import json
        print(json.dumps(extract_url_features("https://www.example-login.com/a?b=1", live_network=False), indent=2))
