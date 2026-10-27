#!/usr/bin/env python3
"""Stage 3 - metadata profiling and related-video identification (CPU, scikit-learn TF-IDF, no network).

Reads 02_eligibility/eligible_candidates.csv (+ 01_consolidation/suspected_reuploads.csv); profiles every
non-excluded candidate and writes <output_root>/03_profile/:
  metadata_profiles.csv          all original columns + station_id, series_id, topic_cluster, region, quarter,
                                 duration_band, relation counts
  programme_families.csv         recurring programmes (same station + normalised title evidence)
  related_video_pairs.csv        potential hard-negative relations: same_series, same_station_near_date,
                                 similar_topic, same_event, possible_reupload (one row per pair, several types allowed)
  metadata_diversity_report.json distributions (categories, overlap, stations, families, topics, geography, dates, durations)
  cache/                         TF-IDF vectoriser, sparse matrix, identifier order (reused when inputs are unchanged)

  python collection/selection/03_profile_metadata.py
  python collection/selection/03_profile_metadata.py --rebuild-features
Relations are POTENTIAL retrieval distractors from metadata, not verified hard negatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from collections import Counter, defaultdict

import numpy as np

import common as C
import metrics as M

URL_RX = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+|@\w+")
SENT_RX = re.compile(r"(?<=[.!?])\s+|\n+|\s{2,}")


# ------------------------------------------------------------------ stations + geography
def stations(rows):
    """station_id per row. Rows without a creator inherit it when their identifier prefix maps to one creator."""
    pref = defaultdict(Counter)
    for r in rows:
        if (r.get("creator") or "").strip():
            pref[C.fold(r.get("station"))][r["creator"].strip()] += 1
    for r in rows:
        key = r.get("channel_key") or C.channel_key(r.get("creator"), r.get("station"))
        if key.startswith("p:"):
            cands = pref.get(C.fold(r.get("station")))
            if cands and len(cands) == 1:
                key = C.channel_key(next(iter(cands)), r.get("station"))
                r["station_inferred_from_prefix"] = 1
        r["channel_key"] = key
    names = defaultdict(Counter)
    for r in rows:
        names[r["channel_key"]][(r.get("creator") or "").strip() or (r.get("station") or "")] += 1
    for r in rows:
        k = r["channel_key"]
        slug = re.sub(r"[^a-z0-9]+", "_", k[2:])[:32].strip("_") or "unknown"
        r["station_id"] = f"st_{slug}_{hashlib.sha1(k.encode()).hexdigest()[:4]}"
        r["station_name"] = names[k].most_common(1)[0][0]


def geography(rows):
    """region from IA subject tags (source metadata), else the station's majority subject-tag region,
    else inferred from a ', ST' suffix in the station name. The broadcaster's location is not the filming location."""
    by_st = defaultdict(Counter)
    for r in rows:
        regs = C.regions_from_subjects(r.get("subjects"))
        r["_regions"] = regs
        for reg in regs:
            by_st[r["station_id"]][reg] += 1
    for r in rows:
        if len(r["_regions"]) == 1:
            (co, rg), src = r["_regions"][0], "ia_subject_tag"
        elif len(r["_regions"]) > 1:
            (co, rg), src = r["_regions"][0], "ia_subject_tag_multiple"
        elif by_st.get(r["station_id"]):
            (co, rg), src = by_st[r["station_id"]].most_common(1)[0][0], "station_majority_subject_tag"
        elif C.region_from_creator(r.get("creator")):
            (co, rg), src = C.region_from_creator(r.get("creator")), "inferred_from_station_name"
        else:
            co, rg, src = "", "", "none"
        r.update(country=co, region=rg, region_source=src,
                 region_reliability={"ia_subject_tag": "source", "ia_subject_tag_multiple": "source_ambiguous",
                                     "station_majority_subject_tag": "source_station_level",
                                     "inferred_from_station_name": "inferred", "none": ""}[src])


# ------------------------------------------------------------------ programme families
def families(rows, pc):
    """Assign series_id from (station, normalised title) evidence, strongest first:
    episode_marker (E153 / Episode 12 / Part 2 before the marker), recurring_title_template (same title once
    dates / numbers are removed, with different date/number tokens), shared_title_prefix (>= 3 videos)."""
    mint = pc["family_min_key_tokens"]
    info = {}
    for r in rows:
        pre, marker = C.title_prefix(r.get("title"))
        info[r["identifier"]] = (pre, marker, C.title_key(r.get("title")), C.episode_tokens(r.get("title")))
    assigned, fams = {}, {}

    def add_groups(keyfn, evidence, conf, min_size, need_distinct_tokens):
        groups = defaultdict(list)
        for r in rows:
            i = r["identifier"]
            if i in assigned:
                continue
            k = keyfn(i)
            if k and len(k.split()) >= mint:
                groups[(r["station_id"], k)].append(r)
        for (st, k), members in groups.items():
            if len(members) < min_size:
                continue
            if need_distinct_tokens and len({info[m["identifier"]][3] for m in members}) < 2:
                continue
            sid = "ser_" + hashlib.sha1(f"{st}|{k}".encode()).hexdigest()[:10]
            for m in members:
                assigned[m["identifier"]] = sid
            dates = sorted(d for d in (C.full_date(m.get("date")) for m in members) if d)
            fams[sid] = {"series_id": sid, "station_id": st, "station_name": members[0]["station_name"],
                         "series_key": k, "evidence": evidence, "confidence": conf, "n_videos": len(members),
                         "first_date": dates[0] if dates else "", "last_date": dates[-1] if dates else "",
                         "categories": dict(Counter(c for m in members for c in C.split_list(m.get("categories")))),
                         "members": [m["identifier"] for m in sorted(members, key=lambda x: (x.get("date") or "", x["identifier"]))],
                         "example_titles": [m.get("title") for m in members[:3]]}
    add_groups(lambda i: info[i][0] if info[i][1] else "", "episode_marker", "high", 2, False)
    add_groups(lambda i: info[i][2], "recurring_title_template", "high", 2, True)
    add_groups(lambda i: info[i][2], "identical_normalised_title", "medium", 2, False)
    add_groups(lambda i: info[i][0], "shared_title_prefix", "medium", pc.get("prefix_family_min_size", 3), False)
    for r in rows:
        sid = assigned.get(r["identifier"], "")
        f = fams.get(sid)
        r.update(series_id=sid, series_confidence=f["confidence"] if f else "", series_evidence=f["evidence"] if f else "",
                 series_size=f["n_videos"] if f else 0, series_key=f["series_key"] if f else "")
    return list(fams.values())


# ------------------------------------------------------------------ text features
def documents(rows, pc):
    """title (twice) + description with URLs and station boilerplate removed (sentences repeated across
    >= boilerplate_min_videos videos of one station, or across >= boilerplate_global_videos videos overall)."""
    sents = {}
    st_count, gl_count = defaultdict(Counter), Counter()
    for r in rows:
        ss = [C.fold(x) for x in SENT_RX.split(URL_RX.sub(" ", r.get("description") or "")) if len(C.fold(x)) > 15]
        sents[r["identifier"]] = ss
        for s in set(ss):
            st_count[r["station_id"]][s] += 1
            gl_count[s] += 1
    docs, removed = [], 0
    for r in rows:
        keep = []
        for s in sents[r["identifier"]]:
            if st_count[r["station_id"]][s] >= pc["boilerplate_min_videos"] or gl_count[s] >= pc["boilerplate_global_videos"]:
                removed += 1
            else:
                keep.append(s)
        t = C.fold(r.get("title"))
        docs.append(f"{t} {t} " + " ".join(keep))
    return docs, removed


def features(rows, docs, pc, seed, cache_dir, rebuild):
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    tf = pc["tfidf"]
    params = {"tfidf": tf, "seed": seed}
    key = hashlib.sha256((json.dumps(params, sort_keys=True) + "\n".join(r["identifier"] + "\t" + d for r, d in
                                                                        zip(rows, docs))).encode()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_p = cache_dir / "features_meta.json"
    if not rebuild and meta_p.exists() and C.read_json(meta_p).get("key") == key:
        C.log("TF-IDF features: cache hit")
        X = sparse.load_npz(cache_dir / "tfidf_matrix.npz")
        with open(cache_dir / "tfidf_vectorizer.pkl", "rb") as f:
            vec = pickle.load(f)
        return X, vec
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, tf["ngram_max"]), min_df=tf["min_df"],
                          max_df=tf["max_df"], max_features=tf["max_features"], sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-z][a-z]+\b", dtype=np.float32)
    X = vec.fit_transform(docs).tocsr()
    sparse.save_npz(cache_dir / "tfidf_matrix.npz", X)
    with open(cache_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)
    C.write_json(cache_dir / "tfidf_identifiers.json", [r["identifier"] for r in rows])
    C.write_json(meta_p, {"key": key, "params": params, "n_docs": X.shape[0], "n_features": X.shape[1],
                          "created": C.now_iso()})
    C.log(f"TF-IDF features: {X.shape[0]} documents x {X.shape[1]} terms (cached in {cache_dir})")
    return X, vec


def topics(X, vec, pc, seed):
    """LSA (TruncatedSVD) + k-means topic clusters, labelled by their highest mean TF-IDF terms."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    nz = np.asarray(X.getnnz(axis=1)).ravel() > 0
    labels = np.full(X.shape[0], -1)
    k = min(pc["topic_clusters"], int(nz.sum()) // 5 or 1)
    if nz.sum() >= k * 2 and X.shape[1] > 2:
        comp = min(pc["svd_components"], X.shape[1] - 1)
        Z = normalize(TruncatedSVD(n_components=comp, random_state=seed).fit_transform(X[nz]))
        labels[nz] = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    terms = np.array(vec.get_feature_names_out())
    info = {}
    for c in sorted(set(labels) - {-1}):
        idx = np.where(labels == c)[0]
        mean = np.asarray(X[idx].mean(axis=0)).ravel()
        info[int(c)] = {"topic_cluster": f"t{int(c):02d}", "n_videos": int(len(idx)),
                        "top_terms": list(terms[np.argsort(mean)[::-1][:8]])}
    return labels, info


def neighbours(X, k):
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    kk = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=kk, metric="cosine", algorithm="brute").fit(X)
    dist, idx = nn.kneighbors(X)
    return 1.0 - dist, idx


# ------------------------------------------------------------------ relations
def relations(rows, X, sims, nbr, fams, reup, pc):
    pos = {r["identifier"]: n for n, r in enumerate(rows)}
    pairs = defaultdict(lambda: {"types": [], "evidence": []})

    def add(a, b, t, ev):
        if a == b:
            return
        key = tuple(sorted((a, b)))
        if t not in pairs[key]["types"]:
            pairs[key]["types"].append(t)
            pairs[key]["evidence"].append(ev)
    for f in fams:
        m = f["members"]
        for x in range(len(m)):
            for y in range(x + 1, len(m)):
                add(m[x], m[y], "same_series", f"{f['series_id']} ({f['evidence']}, {f['confidence']})")
    by_st = defaultdict(list)
    for r in rows:
        d = C.full_date(r.get("date"))
        if d:
            by_st[r["station_id"]].append((d, r["identifier"]))
    for st, lst in by_st.items():
        lst.sort()
        for x in range(len(lst)):
            for y in range(x + 1, len(lst)):
                gap = (lst[y][0] - lst[x][0]).days
                if gap > pc["near_date_days"]:
                    break
                add(lst[x][1], lst[y][1], "same_station_near_date", f"same station, {gap} day(s) apart")
    for i, r in enumerate(rows):
        for s, j in zip(sims[i], nbr[i]):
            if j == i or s <= 0:
                continue
            b = rows[j]
            if s >= pc["similar_topic_threshold"]:
                add(r["identifier"], b["identifier"], "similar_topic", f"title/description TF-IDF cosine {s:.3f}")
            da, db = C.full_date(r.get("date")), C.full_date(b.get("date"))
            if (s >= pc["same_event_threshold"] and da and db and abs((da - db).days) <= pc["same_event_days"]
                    and not (r.get("series_id") and r.get("series_id") == b.get("series_id"))):
                add(r["identifier"], b["identifier"], "same_event",
                    f"TF-IDF cosine {s:.3f} and dates {abs((da - db).days)} day(s) apart")
    for p in reup:
        if p["identifier_a"] in pos and p["identifier_b"] in pos:
            add(p["identifier_a"], p["identifier_b"], "possible_reupload", p.get("evidence", ""))
    out = []
    for (a, b), v in sorted(pairs.items()):
        ra, rb = rows[pos[a]], rows[pos[b]]
        da, db = C.full_date(ra.get("date")), C.full_date(rb.get("date"))
        cos = float(X[pos[a]].multiply(X[pos[b]]).sum())
        out.append({"identifier_a": a, "identifier_b": b, "relation_types": v["types"],
                    "series_id": ra["series_id"] if ra.get("series_id") and ra["series_id"] == rb.get("series_id") else "",
                    "same_station": int(ra["station_id"] == rb["station_id"]),
                    "days_apart": abs((da - db).days) if da and db else "",
                    "topic_cosine": round(cos, 4), "evidence": " | ".join(v["evidence"])})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--rebuild-features", action="store_true", help="ignore the TF-IDF cache")
    a = ap.parse_args(argv)
    cfg = C.load_config(a.config)
    pc, seed = cfg["profiling"], int(cfg["seed"])
    src = C.require(C.stage_dir(cfg, 2, a.out_root) / "eligible_candidates.csv", "stage-2 output")
    reup_p = C.stage_dir(cfg, 1, a.out_root) / "suspected_reuploads.csv"
    out = C.stage_dir(cfg, 3, a.out_root)
    allrows = C.read_csv(src)
    C.validate_columns(allrows, ["identifier", "title", "eligibility_status", "categories", "channel_key"], str(src))
    rows = sorted([r for r in allrows if r["eligibility_status"] != "excluded"], key=lambda r: r["identifier"])
    if not rows:
        raise C.StageError("no eligible or conditional candidates")
    C.log(f"profiling {len(rows)} candidates (eligible + conditional)")
    stations(rows)
    geography(rows)
    fams = families(rows, pc)
    docs, removed = documents(rows, pc)
    X, vec = features(rows, docs, pc, seed, out / "cache", a.rebuild_features)
    labels, tinfo = topics(X, vec, pc, seed)
    sims, nbr = neighbours(X, pc["neighbours_k"])
    for n, r in enumerate(rows):
        r["topic_cluster"] = tinfo[int(labels[n])]["topic_cluster"] if labels[n] >= 0 else "none"
        r["topic_terms"] = tinfo[int(labels[n])]["top_terms"][:5] if labels[n] >= 0 else ""
        mins = C.to_float(r.get("duration_min"))
        r["duration_band"] = C.duration_band(mins, pc["duration_bands"])
        r["year"] = C.year_of(r.get("date")) or ""
        r["quarter"] = C.quarter_of(r.get("date"))
        r["date_exact"] = int(C.full_date(r.get("date")) is not None)
        j = [(x, sv) for x, sv in zip(nbr[n], sims[n]) if x != n and sv > 0]
        r["top_topic_neighbour"] = rows[j[0][0]]["identifier"] if j else ""
        r["top_topic_similarity"] = round(float(j[0][1]), 4) if j else ""
    reup = C.read_csv(reup_p) if reup_p.exists() else []
    pairs = relations(rows, X, sims, nbr, fams, reup, pc)
    cnt = defaultdict(Counter)
    for p in pairs:
        for t in C.split_list(p["relation_types"]):
            cnt[p["identifier_a"]][t] += 1
            cnt[p["identifier_b"]][t] += 1
    rel_types = ["same_series", "same_station_near_date", "similar_topic", "same_event", "possible_reupload"]
    for r in rows:
        for t in rel_types:
            r[f"n_{t}"] = cnt[r["identifier"]][t]
        r.pop("_regions", None)

    base = list(allrows[0].keys())
    added = ["station_id", "station_name", "station_inferred_from_prefix", "country", "region", "region_source",
             "region_reliability", "series_id", "series_key", "series_confidence", "series_evidence", "series_size",
             "topic_cluster", "topic_terms", "top_topic_neighbour", "top_topic_similarity", "duration_band", "year",
             "quarter", "date_exact"] + [f"n_{t}" for t in rel_types]
    C.write_csv(out / "metadata_profiles.csv", rows, base + [f for f in added if f not in base])
    C.write_csv(out / "programme_families.csv", sorted(fams, key=lambda f: (-f["n_videos"], f["series_id"])),
                ["series_id", "station_id", "station_name", "series_key", "evidence", "confidence", "n_videos",
                 "first_date", "last_date", "categories", "members", "example_titles"])
    C.write_csv(out / "related_video_pairs.csv", pairs,
                ["identifier_a", "identifier_b", "relation_types", "series_id", "same_station", "days_apart",
                 "topic_cosine", "evidence"])
    C.write_json(out / "topic_clusters.json", list(tinfo.values()))

    cats = C.CATEGORIES
    co = {a_: {b_: sum(1 for r in rows if int(r.get(f"cat_{a_}") or 0) and int(r.get(f"cat_{b_}") or 0)) for b_ in cats}
          for a_ in cats}
    st = Counter(r["station_id"] for r in rows)
    fsz = Counter(f["n_videos"] for f in fams)
    report = {
        "stage": "03_profile", "created": C.now_iso(), "input": str(src), "input_sha256": C.sha256_file(src),
        "videos_profiled": len(rows),
        "set_metrics": M.set_metrics(rows, pairs),
        "categories": {"counts": {c: co[c][c] for c in cats}, "co_occurrence": co,
                       "labels_per_video": dict(sorted(Counter(int(r["n_categories"] or 0) for r in rows).items()))},
        "stations": {**C.concentration(dict(st)), "top20": [
            {"station_id": k, "station_name": next(r["station_name"] for r in rows if r["station_id"] == k), "videos": v}
            for k, v in st.most_common(20)],
            "inferred_from_identifier_prefix": sum(1 for r in rows if r.get("station_inferred_from_prefix"))},
        "programme_families": {"families": len(fams), "videos_in_families": sum(f["n_videos"] for f in fams),
                               "by_evidence": dict(Counter(f["evidence"] for f in fams)),
                               "by_confidence": dict(Counter(f["confidence"] for f in fams)),
                               "size_distribution": dict(sorted(fsz.items())),
                               "largest": [{k: f[k] for k in ("series_id", "station_name", "series_key", "n_videos")}
                                           for f in sorted(fams, key=lambda f: -f["n_videos"])[:15]]},
        "topics": {"clusters": len(tinfo), "unclustered_videos": int((labels < 0).sum()),
                   "boilerplate_sentences_removed": removed, "tfidf_features": int(X.shape[1]),
                   "clusters_detail": list(tinfo.values())},
        "geography": {"by_source": dict(Counter(r["region_source"] for r in rows)),
                      "countries": dict(Counter(r["country"] or "unknown" for r in rows)),
                      "regions": dict(Counter(r["region"] or "unknown" for r in rows).most_common()),
                      "note": "station / upload location from IA metadata; not necessarily where the footage was filmed"},
        "dates": {"years": dict(sorted(Counter(str(r["year"] or "unknown") for r in rows).items())),
                  "quarters": dict(sorted(Counter(r["quarter"] for r in rows).items())),
                  "exact_dates": sum(r["date_exact"] for r in rows),
                  "upload_date_used": sum(1 for r in rows if r.get("date_source") == "uploaded")},
        "durations": {"bands": dict(Counter(r["duration_band"] for r in rows)),
                      "total_hours": round(sum(C.to_float(r.get("duration_min"), 0) for r in rows) / 60, 1)},
        "relations": {"pairs": len(pairs),
                      "by_type": dict(Counter(t for p in pairs for t in C.split_list(p["relation_types"]))),
                      "pairs_with_multiple_types": sum(1 for p in pairs if len(C.split_list(p["relation_types"])) > 1),
                      "note": "potential retrieval distractors from metadata, not verified hard negatives"},
        "parameters": pc,
    }
    C.write_json(out / "metadata_diversity_report.json", report)
    C.log(f"{len(st)} stations, {len(fams)} programme families ({report['programme_families']['videos_in_families']} "
          f"videos), {len(tinfo)} topic clusters, {len(pairs)} related pairs {report['relations']['by_type']}")
    C.log(f"wrote {out}")


if __name__ == "__main__":
    main()
