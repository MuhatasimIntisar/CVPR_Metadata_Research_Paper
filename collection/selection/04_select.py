#!/usr/bin/env python3
"""Stage 4 - deterministic selection of exactly N videos (default 1,000) by greedy marginal gain.

Reads 03_profile/{metadata_profiles.csv, related_video_pairs.csv} (+ 02_eligibility for excluded candidates) and
writes <output_root>/04_selection/:
  selected_1000.csv         the final strategy's selection, in pick order, with every scoring component
  reserve_candidates.csv    next-best candidates (replacements), ranked by marginal gain w.r.t. the final selection
  selection_decisions.csv   every candidate: selected / reserve / not_selected / excluded + recorded reasons
  selection_comparison.csv  the four strategies side by side
  selection_report.json     configuration, seed, feasibility checks, per-strategy statistics

  python collection/selection/04_select.py
  python collection/selection/04_select.py --budget-hours 500 --target 1000

Objective of one candidate i given the current selection S (all terms recorded per pick):
  coverage   scale * sum_f W_f / Z_f * sum_{a in attrs_f(i)} [log(2 + n_a) - log(1 + n_a)]
             families f: category (multi-label), topic, station, region, date (quarter), duration band;
             Z_f = V_f * log(1 + N / V_f) normalises each family by its number of values V_f;
             scale = N / sum_f W_f, so an average pick is worth ~1
  captions   + caption_bonus * coverage if the video has captions (so captions never outweigh diversity late on)
  series     + series_bonus[k] for the (k+1)-th selected episode of a confidently identified programme family
             (0 for the first episode and beyond the list, so dozens of episodes are never rewarded)
  related    + sum of related_bonus[type] over already-selected partners (same_station_near_date, same_event,
             similar_topic; the strongest type per pair; similar_topic alone capped at similar_topic_cap),
             capped at related_bonus_cap
Hard constraints: exactly N distinct videos; pilot / duplicate / ineligible candidates are not in the pool;
total hours <= budget_hours (checked with the shortest possible completion at every pick); at least
min_per_category videos per category (multi-label); never both videos of a possible_reupload pair (configurable).
If a constraint is infeasible the script stops and says why; nothing is relaxed silently.
"""
from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict

import numpy as np

import common as C
import metrics as M

FAMILIES = ["category", "topic", "station", "region", "date", "duration"]
STRATEGIES = {
    "stratified_random": {"greedy": False},
    "coverage_only": {"greedy": True, "related": False, "budget": False},
    "coverage_related": {"greedy": True, "related": True, "budget": False},
    "coverage_related_budget": {"greedy": True, "related": True, "budget": True},
}


class Infeasible(C.StageError):
    pass


def attrs_of(r):
    a = {"category": [f"cat:{c}" for c in C.split_list(r.get("categories"))],
         "topic": [f"topic:{r['topic_cluster']}"] if r.get("topic_cluster") not in ("", "none") else [],
         "station": [f"st:{r['station_id']}"],
         "region": [f"reg:{r['country']}-{r['region']}"] if r.get("region") else [],
         "date": [f"q:{r['quarter']}"] if r.get("quarter") not in ("", "unknown") else [],
         "duration": [f"dur:{r['duration_band']}"] if r.get("duration_band") not in ("", "unknown") else []}
    return a


class Selector:
    def __init__(self, rows, pairs, sc, seed, related=True, budget=True):
        self.rows = rows
        self.n = len(rows)
        self.budget_block = {}
        self.sc, self.related, self.use_budget = sc, related, budget
        self.N = int(sc["target"])
        self.ids = [r["identifier"] for r in rows]
        self.pos = {i: k for k, i in enumerate(self.ids)}
        self.dur = np.array([C.to_float(r.get("duration_min"), 0.0) / 60 for r in rows])
        self.caps = np.array([1.0 if r.get("has_captions") == "yes" else 0.0 for r in rows])
        self.A = [attrs_of(r) for r in rows]
        W = sc["weights"]
        values = {f: {a for x in self.A for a in x[f]} for f in FAMILIES}
        self.coef = {}
        for f in FAMILIES:
            V = max(1, len(values[f]))
            Z = V * math.log(1 + self.N / V)
            self.coef[f] = W.get(f, 0) / Z if Z else 0
        self.scale = self.N / max(1e-9, sum(W.get(f, 0) for f in FAMILIES))
        self.cats = list(sc.get("categories") or C.CATEGORIES)
        self.cat_of = [set(C.split_list(r.get("categories"))) for r in rows]
        conf_ok = set(sc.get("series_confidence", ["high"]))
        self.series = [r["series_id"] if r.get("series_id") and r.get("series_confidence") in conf_ok else "" for r in rows]
        rb = sc["related_bonus"]
        self.partners = defaultdict(dict)        # k -> {j: weight}
        self.reup = defaultdict(set)
        for p in pairs:
            a, b = self.pos.get(p["identifier_a"]), self.pos.get(p["identifier_b"])
            if a is None or b is None:
                continue
            types = C.split_list(p["relation_types"])
            if "possible_reupload" in types:
                self.reup[a].add(b)
                self.reup[b].add(a)
            strong = max([rb.get(t, 0) for t in types if t in rb and t != "similar_topic"] or [0])
            w, kind = (strong, "strong") if strong > 0 else (rb.get("similar_topic", 0) if "similar_topic" in types else 0, "topic")
            if w > 0:
                self.partners[a][b] = (w, kind)
                self.partners[b][a] = (w, kind)
        rng = random.Random(seed)
        self.tieb = {k: rng.random() for k in range(self.n)}
        self.reset()

    def reset(self):
        self.count = Counter()
        self.sel, self.order, self.series_n = set(), [], Counter()
        self.hours = 0.0
        self.cat_n = Counter()
        self.log = {}

    # -------------------------------------------------- scoring
    def parts(self, k):
        cov = {}
        for f in FAMILIES:
            g = 0.0
            for a in self.A[k][f]:
                n = self.count[a]
                g += math.log(2 + n) - math.log(1 + n)
            cov[f] = self.scale * self.coef[f] * g
        cap = self.sc["caption_bonus"] * self.caps[k] * sum(cov.values())  # captions scale the coverage value
        ser = rel = 0.0
        if self.related:
            s = self.series[k]
            if s:
                b = self.sc["series_bonus"]
                m = self.series_n[s]
                ser = b[m] if m < len(b) else 0.0
            strong = sum(w for j, (w, kd) in self.partners[k].items() if kd == "strong" and j in self.sel)
            topic = sum(w for j, (w, kd) in self.partners[k].items() if kd == "topic" and j in self.sel)
            rel = min(self.sc["related_bonus_cap"], strong + min(self.sc.get("similar_topic_cap", 0.2), topic))
        return cov, cap, ser, rel

    def gain(self, k):
        cov, cap, ser, rel = self.parts(k)
        return sum(cov.values()) + cap + ser + rel

    def add(self, k, phase, keep_log=None):
        cov, cap, ser, rel = self.parts(k)
        self.log[k] = {"rank": len(self.order) + 1, "phase": phase, "gain_total": sum(cov.values()) + cap + ser + rel,
                       **{f"gain_{f}": v for f, v in cov.items()}, "gain_coverage": sum(cov.values()),
                       "gain_captions": cap, "gain_series": ser, "gain_related": rel,
                       "hours_after": self.hours + self.dur[k]}
        if keep_log is not None:          # re-adding during local search: keep the original record
            self.log[k] = keep_log
        for f in FAMILIES:
            for a in self.A[k][f]:
                self.count[a] += 1
        if self.series[k]:
            self.series_n[self.series[k]] += 1
        for c in self.cat_of[k]:
            self.cat_n[c] += 1
        self.sel.add(k)
        self.order.append(k)
        self.hours += self.dur[k]

    def remove(self, k):
        for f in FAMILIES:
            for a in self.A[k][f]:
                self.count[a] -= 1
        if self.series[k]:
            self.series_n[self.series[k]] -= 1
        for c in self.cat_of[k]:
            self.cat_n[c] -= 1
        self.sel.discard(k)
        self.order.remove(k)
        self.hours -= self.dur[k]

    # -------------------------------------------------- constraints
    def blocked(self, k):
        if k in self.sel:
            return "selected"
        if self.sc.get("avoid_possible_reupload_pairs", True) and self.reup[k] & self.sel:
            return "possible_reupload_of_selected"
        return ""

    def budget_mask(self, cand):
        """Candidates whose pick still leaves room to finish with the shortest remaining videos."""
        if not self.use_budget:
            return np.ones(len(cand), bool)
        B = self.sc["budget_hours"]
        rest = self.N - len(self.sel) - 1
        free = np.array([k for k in range(self.n) if k not in self.sel])
        d = np.sort(self.dur[free])
        pre = np.concatenate([[0.0], np.cumsum(d)])
        thr = d[rest - 1] if rest > 0 else -1.0
        out = []
        for k in cand:
            if rest <= 0:
                need = 0.0
            elif self.dur[k] <= thr:  # k is among the shortest: replace it by the next one
                need = pre[min(rest + 1, len(d))] - self.dur[k]
            else:
                need = pre[rest]
            out.append(self.hours + self.dur[k] + need <= B + 1e-9)
        return np.array(out, bool)

    def check_feasible(self):
        msgs = []
        if self.n < self.N:
            msgs.append(f"pool has {self.n} usable videos < target {self.N}")
        mins = self.sc["min_per_category"]
        avail = Counter(c for s in self.cat_of for c in s)
        for c in self.cats:
            if avail[c] < mins:
                msgs.append(f"category {c}: only {avail[c]} candidates < min_per_category {mins}")
        if self.use_budget:
            shortest = float(np.sort(self.dur)[:self.N].sum())
            if shortest > self.sc["budget_hours"]:
                msgs.append(f"budget {self.sc['budget_hours']} h infeasible: the {self.N} shortest videos already "
                            f"total {shortest:.1f} h")
        if msgs:
            raise Infeasible("; ".join(msgs))

    def deficits(self):
        m = self.sc["min_per_category"]
        return {c: m - self.cat_n[c] for c in self.cats if self.cat_n[c] < m}

    # -------------------------------------------------- greedy
    def run(self):
        self.check_feasible()
        self.reset()
        self.budget_block = {}                 # k -> pick number at which the hours budget first ruled it out
        while len(self.sel) < self.N:
            deficit = self.deficits()
            cand = [k for k in range(self.n) if not self.blocked(k)]
            # category minimums are enforced only when the remaining slots are just enough to reach them
            # (each restricted pick lowers the total deficit by >= 1, so this always stays feasible)
            tight = deficit and self.N - len(self.sel) <= sum(deficit.values())
            phase = "category_minimum" if tight else "free"
            if tight:
                c2 = [k for k in cand if self.cat_of[k] & set(deficit)]
                cand = c2 or cand
            if not cand:
                raise Infeasible(f"no feasible candidate left after {len(self.sel)} picks (re-upload conflicts)")
            ok = self.budget_mask(cand)
            for k, o in zip(cand, ok):
                if not o and k not in self.budget_block:
                    self.budget_block[k] = len(self.sel) + 1
            cand = [k for k, o in zip(cand, ok) if o]
            if not cand:
                raise Infeasible(f"budget {self.sc['budget_hours']} h exhausted after {len(self.sel)} picks "
                                 f"({self.hours:.1f} h used)")
            best = max(cand, key=lambda k: (self.gain(k), self.tieb[k]))
            self.add(best, phase)
        if self.deficits():
            raise Infeasible(f"category minimums not reached: {self.deficits()}")
        return self.order

    def swaps(self, passes, m):
        """Bounded local search: try replacing the m lowest-contribution selected videos by the best unselected
        one; accept a swap only if it raises the objective and keeps every hard constraint."""
        done = []
        mins = self.sc["min_per_category"]
        B = self.sc["budget_hours"]
        for _ in range(passes):
            improved = False
            loss = {}
            for k in list(self.sel):
                saved = self.log[k]
                self.remove(k)
                loss[k] = self.gain(k)
                self.add(k, saved["phase"], keep_log=saved)
            worst = sorted(loss, key=lambda k: (loss[k], self.tieb[k]))[:m]
            for i in worst:
                if i not in self.sel:
                    continue
                saved = self.log[i]
                self.remove(i)
                li = self.gain(i)
                pool = [k for k in range(self.n) if k != i and not self.blocked(k)
                        and all(self.cat_n[c] + (c in self.cat_of[k]) >= mins for c in self.cats)
                        and (not self.use_budget or self.hours + self.dur[k] <= B + 1e-9)]
                best = max(pool, key=lambda k: (self.gain(k), self.tieb[k]), default=None)
                if best is not None and self.gain(best) > li + 1e-6:
                    g = self.gain(best)
                    self.add(best, "swap")
                    self.log[best]["rank"] = saved["rank"]
                    self.log[best]["replaced"] = self.ids[i]
                    self.log.pop(i, None)
                    done.append({"out": self.ids[i], "in": self.ids[best], "gain_in": round(g, 5), "loss_out": round(li, 5)})
                    improved = True
                else:
                    self.add(i, saved["phase"], keep_log=saved)
            if not improved:
                break
        self.order.sort(key=lambda k: self.log[k]["rank"])
        return done

    def objective(self):
        """Value of the current selection: the gains obtained when its videos are added in rank order."""
        order, state = list(self.order), (self.count, self.sel, self.order, self.series_n, self.hours, self.cat_n, self.log)
        self.reset()
        total = 0.0
        for k in order:
            total += self.gain(k)
            self.add(k, "replay")
        self.count, self.sel, self.order, self.series_n, self.hours, self.cat_n, self.log = state
        return round(total, 4)


def stratified_random(rows, N, seed, strat_key="primary_category"):
    """Seeded random sample with equal allocation per primary (discovery) category; spare slots are refilled
    proportionally from strata that still have candidates."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for k, r in enumerate(rows):
        strata[r.get(strat_key) or "unknown"].append(k)
    for v in strata.values():
        rng.shuffle(v)
    names = sorted(strata)
    quota = {s: min(len(strata[s]), N // len(names)) for s in names}
    while sum(quota.values()) < N and any(quota[s] < len(strata[s]) for s in names):
        for s in names:
            if sum(quota.values()) < N and quota[s] < len(strata[s]):
                quota[s] += 1
    return [k for s in names for k in strata[s][:quota[s]]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--target", type=int, help="override selection.target")
    ap.add_argument("--budget-hours", type=float, help="override selection.budget_hours")
    ap.add_argument("--strategy", choices=list(STRATEGIES), help="override selection.final_strategy")
    ap.add_argument("--no-swaps", action="store_true", help="skip the local-swap refinement")
    a = ap.parse_args(argv)
    cfg = C.load_config(a.config)
    sc = dict(cfg["selection"])
    if a.target:
        sc["target"] = a.target
    if a.budget_hours:
        sc["budget_hours"] = a.budget_hours
    final = a.strategy or sc["final_strategy"]
    seed = int(cfg["seed"])
    d3 = C.stage_dir(cfg, 3, a.out_root)
    prof_p = C.require(d3 / "metadata_profiles.csv", "stage-3 profiles")
    pairs_p = C.require(d3 / "related_video_pairs.csv", "stage-3 related pairs")
    elig_p = C.require(C.stage_dir(cfg, 2, a.out_root) / "eligible_candidates.csv", "stage-2 output")
    out = C.stage_dir(cfg, 4, a.out_root)
    rows = sorted(C.read_csv(prof_p), key=lambda r: r["identifier"])
    C.validate_columns(rows, ["identifier", "categories", "station_id", "topic_cluster", "duration_min", "series_id",
                              "quarter", "duration_band", "has_captions"], str(prof_p))
    pairs = C.read_csv(pairs_p)
    N = int(sc["target"])
    C.log(f"pool {len(rows)} candidates, {len(pairs)} related pairs; target {N}, budget {sc['budget_hours']} h, "
          f"min {sc['min_per_category']} per category; final strategy {final}")

    results, selectors = {}, {}
    for name, spec in STRATEGIES.items():
        if not spec["greedy"]:
            picks = stratified_random(rows, N, seed)
            results[name] = {"picks": picks, "objective": None, "swaps": []}
            continue
        S = Selector(rows, pairs, sc, seed, related=spec["related"], budget=spec["budget"])
        try:
            S.run()
        except Infeasible as e:
            if name == final:
                raise
            C.log(f"  {name}: infeasible ({e}); left out of the comparison")
            continue
        sw = S.swaps(sc["swap_passes"], sc["swap_candidates"]) if (name == final and not a.no_swaps) else []
        results[name] = {"picks": list(S.order), "objective": S.objective(), "swaps": sw}
        selectors[name] = S
        C.log(f"  {name}: {len(S.order)} picks, {S.hours:.1f} h, objective {S.objective()}"
              + (f", {len(sw)} swaps" if sw else ""))

    comp, mets = [], {}
    for name, res in results.items():
        mrows = [rows[k] for k in res["picks"]]
        mets[name] = M.set_metrics(mrows, pairs, pool_rows=rows)
        comp.append({"strategy": name, "objective": res["objective"], **M.flat(mets[name])})
    C.write_csv(out / "selection_comparison.csv", comp)

    S = selectors[final]
    chosen = set(S.order)
    if len(chosen) != N or len({rows[k]["identifier"] for k in chosen}) != N:
        raise C.StageError(f"internal error: final selection has {len(chosen)} videos, expected {N} distinct")
    # reserve: rank the rest by marginal gain w.r.t. the final selection (budget ignored: reserves are swaps)
    rest = [k for k in range(len(rows)) if k not in chosen]
    rg = {k: S.gain(k) for k in rest}
    conflict = {k: S.blocked(k) for k in rest}
    reserve = [k for k in sorted(rest, key=lambda k: (-rg[k], S.tieb[k])) if not conflict[k]][:sc["reserve_size"]]
    rrank = {k: n + 1 for n, k in enumerate(reserve)}

    sel_rows = []
    for k in S.order:
        r = dict(rows[k])
        r.update({f"sel_{x}": round(v, 5) if isinstance(v, float) else v for x, v in S.log[k].items()},
                 selection_strategy=final)
        sel_rows.append(r)
    base_fields = list(rows[0].keys())
    sel_fields = base_fields + [f for f in dict.fromkeys(x for r in sel_rows for x in r) if f not in base_fields]
    C.write_csv(out / f"selected_{N}.csv", sel_rows, sel_fields)
    res_rows = []
    for k in reserve:
        cov, cap, ser, rel = S.parts(k)
        res_rows.append({**rows[k], "reserve_rank": rrank[k], "reserve_gain_total": round(rg[k], 5),
                         "reserve_gain_coverage": round(sum(cov.values()), 5), "reserve_gain_series": ser,
                         "reserve_gain_related": round(rel, 5)})
    C.write_csv(out / "reserve_candidates.csv", res_rows,
                base_fields + ["reserve_rank", "reserve_gain_total", "reserve_gain_coverage", "reserve_gain_series",
                               "reserve_gain_related"])

    # decisions for every consolidated candidate (including stage-1/2 exclusions)
    dec = []
    for r in C.read_csv(elig_p):
        i = r["identifier"]
        k = S.pos.get(i)
        d = {"identifier": i, "title": r.get("title"), "primary_category": r.get("primary_category"),
             "categories": r.get("categories"), "eligibility_status": r.get("eligibility_status")}
        if k is None:
            d.update(status="excluded", reasons=r.get("eligibility_reasons") or f"consolidation:{r.get('exclusion_reason')}")
        elif k in chosen:
            lg = S.log[k]
            d.update(status="selected", rank=lg["rank"], phase=lg["phase"],
                     reasons=("filled category minimum" if lg["phase"] == "category_minimum" else
                              "replaced " + lg["replaced"] + " in local swap" if lg.get("replaced") else
                              "highest marginal gain at its pick"),
                     **{f"gain_{x}": round(lg[f"gain_{x}"], 5) for x in FAMILIES + ["coverage", "captions", "series", "related", "total"]},
                     constraints=f"budget ok (hours after pick {lg['hours_after']:.1f}); no re-upload conflict")
        else:
            cov, cap, ser, rel = S.parts(k)
            bb = S.budget_block.get(k)
            d.update(status="reserve" if k in rrank else "not_selected", reserve_rank=rrank.get(k, ""),
                     reasons=(f"conflict: {conflict[k]}" if conflict[k] else
                              f"hours budget ruled it out from pick {bb} on ({C.to_float(rows[k].get('duration_min'), 0):.0f} min)"
                              if bb else "never the highest marginal gain among feasible candidates"),
                     **{f"gain_{x}": round(v, 5) for x, v in cov.items()}, gain_coverage=round(sum(cov.values()), 5),
                     gain_captions=cap, gain_series=ser, gain_related=round(rel, 5), gain_total=round(rg[k], 5),
                     constraints=conflict[k] or "")
        d["eligibility_flags"] = r.get("eligibility_flags")
        dec.append(d)
    C.write_csv(out / "selection_decisions.csv", dec,
                ["identifier", "title", "status", "rank", "reserve_rank", "phase", "reasons", "constraints",
                 "primary_category", "categories", "eligibility_status", "eligibility_flags", "gain_total",
                 "gain_coverage"] + [f"gain_{f}" for f in FAMILIES] + ["gain_captions", "gain_series", "gain_related"])

    report = {
        "stage": "04_selection", "created": C.now_iso(), "seed": seed, "final_strategy": final, "selection_config": sc,
        "inputs": {"profiles": str(prof_p), "profiles_sha256": C.sha256_file(prof_p), "pairs": str(pairs_p),
                   "pairs_sha256": C.sha256_file(pairs_p)},
        "code": C.code_version(),
        "pool": {"videos": len(rows), "hours": round(float(sum(C.to_float(r["duration_min"], 0) for r in rows)) / 60, 1),
                 "metrics": M.set_metrics(rows, pairs)},
        "feasibility": {"shortest_N_hours": round(float(np.sort(S.dur)[:N].sum()), 1), "budget_hours": sc["budget_hours"],
                        "category_availability": dict(Counter(c for s in S.cat_of for c in s))},
        "strategies": {n: {"objective": results[n]["objective"], "metrics": mets[n], "swaps": len(results[n]["swaps"])}
                       for n in results},
        "final": {"videos": N, "hours": round(S.hours, 1), "objective": S.objective(),
                  "phases": dict(Counter(S.log[k]["phase"] for k in chosen)), "swaps": results[final]["swaps"],
                  "reserve": len(reserve)},
        "note": "related-video counts are potential retrieval distractors from metadata, not verified hard negatives",
    }
    C.write_json(out / "selection_report.json", report)
    fm = mets[final]
    C.log(f"final {final}: {N} videos, {fm['total_hours']} h, {fm['estimated_storage_gb']} GB, captions "
          f"{fm['caption_share']:.0%}, {fm['stations']['n_unique']} stations, min category {fm['category_min']}, "
          f"{fm['families_with_multiple_selected']} multi-episode families")
    C.log(f"wrote {out}")


if __name__ == "__main__":
    main()
