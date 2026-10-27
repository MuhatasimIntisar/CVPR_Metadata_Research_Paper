"""Set-level statistics shared by stages 3-5 (pool profile, strategy comparison, final report)."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import common as C

STRONG = ("same_series", "same_station_near_date", "same_event", "possible_reupload")


def components(ids, pairs, types=STRONG):
    """Connected components of the related-video graph restricted to `ids` and relation `types`."""
    ids = set(ids)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for p in pairs:
        a, b = p["identifier_a"], p["identifier_b"]
        if a in ids and b in ids and any(t in types for t in C.split_list(p["relation_types"])):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    groups = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    return list(groups.values())


def set_metrics(rows, pairs, pool_rows=None, cats=C.CATEGORIES):
    """rows: profile rows (dicts) of one video set. pairs: related_video_pairs rows."""
    n = len(rows)
    ids = {r["identifier"] for r in rows}
    cat_counts = {c: sum(int(r.get(f"cat_{c}") or 0) for r in rows) for c in cats}
    st = Counter(r["station_id"] for r in rows)
    topics = Counter(r["topic_cluster"] for r in rows if r.get("topic_cluster") not in ("", "none"))
    regions = Counter(r["region"] for r in rows if r.get("region"))
    years = Counter(str(C.year_of(r.get("date")) or "unknown") for r in rows)
    quarters = Counter(r.get("quarter") or "unknown" for r in rows)
    bands = Counter(r.get("duration_band") or "unknown" for r in rows)
    fam = Counter(r["series_id"] for r in rows if r.get("series_id"))
    multi = {k: v for k, v in fam.items() if v >= 2}
    rel = Counter()
    for p in pairs:
        if p["identifier_a"] in ids and p["identifier_b"] in ids:
            for t in C.split_list(p["relation_types"]):
                rel[t] += 1
    comp = [len(g) for g in components(ids, pairs) if len(g) > 1]
    hours = sum(C.to_float(r.get("duration_min"), 0) for r in rows) / 60
    gb = sum(C.to_float(r.get("video_mb"), 0) for r in rows) / 1024
    out = {
        "videos": n,
        "category_counts": cat_counts,
        "category_min": min(cat_counts.values()) if cats else None,
        "categories_covered": sum(1 for v in cat_counts.values() if v),
        "mean_labels_per_video": round(sum(int(r.get("n_categories") or 0) for r in rows) / n, 3) if n else 0,
        "multilabel_videos": sum(1 for r in rows if int(r.get("n_categories") or 0) >= 2),
        "stations": C.concentration(dict(st)),
        "topic_clusters_covered": len(topics),
        "topic_entropy": round(_entropy(topics), 4),
        "regions_covered": len(regions),
        "videos_with_region": sum(regions.values()),
        "years": dict(sorted(years.items())),
        "quarters_covered": len([q for q in quarters if "Q" in q]),
        "duration_bands": dict(sorted(bands.items())),
        "families_with_multiple_selected": len(multi),
        "videos_in_multi_episode_families": sum(multi.values()),
        "related_pairs_retained": dict(rel),
        "related_cluster_sizes": dict(sorted(Counter(comp).items())),
        "caption_share": round(sum(r.get("has_captions") == "yes" for r in rows) / n, 4) if n else 0,
        "total_hours": round(hours, 1),
        "estimated_storage_gb": round(gb, 1),
    }
    if pool_rows is not None:
        pt = Counter(r["topic_cluster"] for r in pool_rows if r.get("topic_cluster") not in ("", "none"))
        out["coverage_vs_pool"] = {
            "stations": round(len(st) / max(1, len({r['station_id'] for r in pool_rows})), 4),
            "topic_clusters": round(len(topics) / max(1, len(pt)), 4),
            "regions": round(len(regions) / max(1, len({r['region'] for r in pool_rows if r.get('region')})), 4),
            "quarters": round(out["quarters_covered"] / max(1, len({r.get('quarter') for r in pool_rows if 'Q' in (r.get('quarter') or '')})), 4)}
    return out


def flat(m: dict, prefix="") -> dict:
    """Nested metrics -> one flat row (for selection_comparison.csv)."""
    row = {}
    for k, v in m.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            row.update(flat(v, key + "."))
        else:
            row[key] = v
    return row


def _entropy(counter):
    tot = sum(counter.values())
    if tot == 0 or len(counter) < 2:
        return 0.0
    return -sum((v / tot) * math.log(v / tot) for v in counter.values()) / math.log(len(counter))
