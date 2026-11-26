"""Single source of truth for the 18 engineered URL feature column names,
so prepare_manifest.py, datasets.py, and fit_url_scaler.py all agree."""

FEATURE_COLUMNS = [
    "entropy", "url_length", "num_www", "ratio_digits_url", "num_subdomains",
    "length_hostname", "random_domain", "num_dots", "num_hyphens",
    "ratio_digits_host", "prefix_suffix", "num_question_mark", "num_eq",
    "domain_age", "dns_resolvable", "num_ips", "ttl", "num_name_servers",
]
