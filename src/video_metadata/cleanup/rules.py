"""Rule-based cleanup of the Level-2 metadata (no models). Every rule edits the document in place and logs each change.

Order (each rule sees the output of the previous ones):
  merge      near-duplicate neighbouring moments of a segment (silent footage) are merged; observed block recomputed
  ocr        on-screen text: glued separators split, spaced letters joined, misreads corrected or dropped, timers collapsed
  entities   people only mentioned are marked transcript-only; places / organisations typed as people re-typed
  names      spelling from on-screen text / archive metadata; variants of one name merged within the video
  appearance a name whose description matches another named person's look is moved or removed; man/woman conflicts;
             description sentences that put a name on the wrong look are flagged
  vocab      dropped relationships mapped back onto the closed predicate list; action synonyms merged
  tags       appearance-only and generic tags removed; single words already inside another tag removed
  shot       shot_scale taken from the pose detector when it is confident and the model disagrees
  speakers   diarisation speakers with almost no talk time in the moment removed
  location   video-level location from the archive subject tags
  segments   single-moment segments marked
  review     review reasons recomputed: false alarms suppressed (kept in review_reasons_suppressed), cleanup reasons added
"""
from __future__ import annotations

import copy
import difflib
import re
from collections import Counter, defaultdict

from . import geo
from .vocab import (ACTION_MAP, FEMALE, MALE, NOT_VISIBLE_RX, ORG_RX, PLACE_RX, PREDICATE_MAP, SCREEN_RX,
                    TAG_DROP_WORDS, WEIGHTS)

NORMALISE_WORD = {"blond": "blonde", "gray": "grey", "t-shirt": "shirt", "tshirt": "shirt", "armour": "armor",
                  "moustache": "mustache", "bearded": "beard"}


class Ctx:
    """Everything the rules read besides the document itself."""

    def __init__(self, cfg, src, transcript, ocr_raw, speakers, observed_fn=None):
        self.c = cfg
        self.src = src or {}
        self.log = []
        meta = " ".join(str(self.src.get(k) or "") for k in ("title", "description", "creator"))
        meta += " " + " ".join(geo._subjects(self.src))
        self.meta_text = re.sub(r"<[^>]+>", " ", meta)
        self.vocab = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z']+", self.meta_text + " " + " ".join(
            s.get("text", "") for s in (transcript or {}).get("segments", [])))}
        self.vocab_list = sorted(self.vocab)
        lines = [it["text"] for smp in (ocr_raw or {}).get("samples", []) for it in smp["items"]]
        self.ocr_tokens = Counter(t.lower() for ln in set(lines) for t in re.findall(r"[A-Za-z0-9]+", ln))
        self.ocr_lines = []  # cleaned, distinct (filled by the ocr rule)
        self._raw_ocr_lines = list(dict.fromkeys(lines))
        self.meta_lines = [x for x in re.split(r"[.\n!?]|<br\s*/?>", str(self.src.get("description") or "")) if x.strip()]
        self.meta_lines += [str(self.src.get("title") or "")] + geo._subjects(self.src)
        self.runs = (speakers or {}).get("runs") or []
        self.observed_fn = observed_fn  # (start, end) -> observed block, for merged moments

    def change(self, m, rule, what, old=None, new=None, why=""):
        rec = {"rule": rule, "moment_id": m["moment_id"] if m else None, "what": what, "from": old, "to": new, "why": why}
        self.log.append(rec)
        if m is not None:
            txt = f"{rule}: {what}" + (f" {old!r} -> {new!r}" if old is not None or new is not None else "") + (f" ({why})" if why else "")
            m["annotation"].setdefault("cleanup_changes", []).append(txt)

    def flag(self, m, reason):
        m.setdefault("_review_add", [])
        if reason not in m["_review_add"]:
            m["_review_add"].append(reason)


def ratio(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def moment_text(m):
    return " ".join([m.get("description") or "", " ".join(m.get("tags") or []), str((m.get("context") or {}).get("setting") or "")])


def is_screen(m):
    return bool(re.search(SCREEN_RX, moment_text(m), re.I))


def persons(m):
    return [e for e in m.get("entities") or [] if e.get("type") == "person"]


# ===================================================================== merge
def merge_duplicates(doc, ctx):
    c = ctx.c["merge"]
    M = doc["moments"]
    out, i = [], 0
    while i < len(M):
        a = M[i]
        while i + 1 < len(M) and _mergeable(a, M[i + 1], c):
            b = M[i + 1]
            a = _merge(a, b, ctx)
            i += 1
        out.append(a)
        i += 1
    if len(out) != len(M):
        doc["moments"] = out
        for s in doc["segments"]:
            s["children"] = [m["moment_id"] for m in out if m["segment_id"] == s["segment_id"]]


def _names(m):
    return {e["name"] for e in m.get("entities") or [] if e.get("name")}


def _mergeable(a, b, c):
    if a["segment_id"] != b["segment_id"] or not a.get("description") or not b.get("description"):
        return False
    if b["end_s"] - a["start_s"] > c["max_merged_s"]:
        return False
    if ratio(a["description"], b["description"]) < c["min_description_sim"]:
        return False
    ta, tb = set(a.get("tags") or []), set(b.get("tags") or [])
    if not ta or not tb or len(ta & tb) / len(ta | tb) < c["min_tag_jaccard"]:
        return False
    aa, ab = a.get("attributes") or {}, b.get("attributes") or {}
    if any(aa.get(k) != ab.get(k) for k in ("shot_scale", "camera_motion")) or _names(a) != _names(b):
        return False
    # only footage without speech: with speech, neighbouring moments differ by what is said even when the picture
    # (and so the description) repeats
    return all((x.get("observed") or {}).get("speech_share", 1) <= c["max_speech_share"] for x in (a, b))


def _merge(a, b, ctx):
    m = copy.deepcopy(a)
    m["end_s"] = b["end_s"]
    m["source_intervals"] = a["source_intervals"] + b["source_intervals"]
    m["boundary"] = {"start": a["boundary"]["start"], "end": b["boundary"]["end"]}
    if len(b["description"]) > len(a["description"]):
        m["description"] = b["description"]
    m["tags"] = list(dict.fromkeys((a.get("tags") or []) + (b.get("tags") or [])))[:12]
    have = {x.get("type") for x in m.get("actions") or []}
    for x in b.get("actions") or []:
        ids = {e["id"] for e in m.get("entities") or []}
        if x.get("type") not in have and set(x.get("participants") or []) <= ids:
            m["actions"].append(copy.deepcopy(x))
    m["merged_from"] = a.get("merged_from", [a["moment_id"]]) + b.get("merged_from", [b["moment_id"]])
    an, bn = m["annotation"], b["annotation"]
    an["review_reasons"] = list(dict.fromkeys(an.get("review_reasons", []) + bn.get("review_reasons", [])))
    an["normalisation_repairs"] = an.get("normalisation_repairs", []) + bn.get("normalisation_repairs", [])
    ep, bp = an.get("evidence_provenance") or {}, bn.get("evidence_provenance") or {}
    ep["frames_s"] = (ep.get("frames_s") or []) + (bp.get("frames_s") or [])
    ep["level1_intervals"] = (ep.get("level1_intervals") or []) + (bp.get("level1_intervals") or [])
    if bn.get("generation_status") != "ok":
        an["generation_status"] = bn.get("generation_status")
    if ctx.observed_fn:
        m["observed"] = ctx.observed_fn(m["start_s"], m["end_s"])
    ctx.change(m, "merge", f"moment {b['moment_id'].rsplit('_', 1)[-1]} merged into this one",
               why=f"near-identical description ({ratio(a['description'], b['description']):.2f}) and tags, no distinguishing speech")
    return m


# ===================================================================== ocr
def _case_like(w, ref):
    return w.upper() if ref.isupper() else (w.title() if ref[:1].isupper() else w)


class OcrCleaner:
    def __init__(self, ctx):
        self.ctx = ctx
        self.fuzzy = ctx.c["ocr"]["fuzzy"]

    def known(self, w, exclude=None):
        w = w.lower()
        return w in self.ctx.vocab or (w != exclude and self.ctx.ocr_tokens.get(w, 0) > 0)

    def word(self, tok, last):
        core = tok.strip(".,;:!?'\"()[]")
        if sum(ch.isalpha() for ch in core) < 4 or core.lower() in self.ctx.vocab:
            return tok
        low = core.lower()
        for p in range(3, len(core) - 1):  # "SoltanilParvazTV" -> "Soltani | ParvazTV" (mixed case only: "MEETING")
            if not core.isupper() and core[p] in "lI|" and core[p + 1].isupper() and self.known(core[:p], exclude=low):
                return tok.replace(core, f"{core[:p]} | {core[p + 1:]}")
        if len(core) >= 5 and core[-1] in "lLIi" and self.known(core[:-1], exclude=low):  # "Societyl KTV", "MOREL", "Royeri"
            return tok.replace(core, core[:-1] + ("" if last else " |"))
        # spelling correction only for tokens that cannot be words (a symbol or digit inside the letters, "De€ember"):
        # without a dictionary, "Please" -> "Pleased" or "COMING" -> "UPCOMING" would be wrong corrections
        if len(core) >= 5 and re.search(r"[A-Za-z][^A-Za-z\s'-]+[A-Za-z]", core):
            best = difflib.get_close_matches(low, self.ctx.vocab_list, n=1, cutoff=self.fuzzy)
            if best and abs(len(best[0]) - len(low)) <= 2:
                return tok.replace(core, _case_like(best[0], core))
        return tok

    def line(self, s):
        s = " ".join(str(s).split())
        s = re.sub(r"(?<!\S)((?:\S ){2,}\S)(?!\S)", lambda m: m.group(1).replace(" ", ""), s)  # "2 0 2 6" -> "2026"
        s = re.sub(r"(?<=\d)\s*:\s*(?=\d)", ":", s)
        s = re.sub(r"(?<=:\d) (?=\d)|(?<=\d) (?=\d:)", "", s)  # "11:17:14:5 9" -> "11:17:14:59"
        toks = s.split(" ")
        s = " ".join(self.word(t, k == len(toks) - 1) for k, t in enumerate(toks))
        s = re.sub(r"\s*\|\s*", " | ", s).strip(" |")
        words = re.findall(r"[A-Za-z]{2,}", s)
        if sum(ch.isalnum() for ch in s) < 4:
            return None
        if len(s.split()) == 1 and words and not any(w.lower() in self.ctx.vocab for w in words):
            # a single unknown word that resembles a known word or another OCR word is a misread or logo fragment
            # ("Royerie" ~ Royer, "Farks" ~ parks, "EATI" ~ "FEATI"); an unrelated one ("Pledge") is kept
            low = re.sub(r"[^a-z0-9]", "", s.lower())
            pool = [w for w in self.ctx.vocab_list if abs(len(w) - len(low)) <= 3] + \
                   [w for w in self.ctx.ocr_tokens if w != low and abs(len(w) - len(low)) <= 3]
            if difflib.get_close_matches(low, pool, n=1, cutoff=0.75):
                return None
        return s


def clean_ocr(doc, ctx):
    cl = OcrCleaner(ctx)
    cache = {}

    def clean(x):
        if x not in cache:
            cache[x] = cl.line(x)
        return cache[x]
    ctx.ocr_lines = [y for y in dict.fromkeys(clean(x) for x in ctx._raw_ocr_lines) if y]
    for m in doc["moments"]:
        ob = m.get("observed") or {}
        old = ob.get("on_screen_text") or []
        new, seen_masks = [], set()
        for x in old:
            y = clean(x)
            if not y:
                continue
            digits = sum(ch.isdigit() for ch in y)
            if digits >= 0.5 * sum(ch.isalnum() for ch in y):  # timers / clocks: keep one reading
                mask = re.sub(r"\d", "#", y)
                if mask in seen_masks:
                    continue
                seen_masks.add(mask)
            if y not in new:
                new.append(y)
        if new != old:
            ob["on_screen_text"] = new
            dropped = [x for x in old if not clean(x)]
            ctx.change(m, "ocr", "on_screen_text cleaned", why=f"{len(old)} -> {len(new)} lines" +
                       (f"; dropped {dropped[:4]}" if dropped else ""))


# ===================================================================== entities
def fix_entities(doc, ctx):
    for m in doc["moments"]:
        for e in m.get("entities") or []:
            if e.get("type") != "person":
                continue
            d, nm = e.get("description") or "", e.get("name") or ""
            if re.search(r"\b(location|place|city|town)\b", d, re.I) and re.search(PLACE_RX, d, re.I):
                ctx.change(m, "entities", f"{e['id']} type", "person", "place", f"description: {d!r}")
                e["type"] = "place"
                continue
            if nm and re.search(ORG_RX, nm, re.I):
                ctx.change(m, "entities", f"{e['id']} type", "person", "organization", f"name {nm!r}")
                e["type"] = "organization"
                continue
            if re.search(NOT_VISIBLE_RX, d, re.I) and e.get("evidence") != "transcript":
                ctx.change(m, "entities", f"{e['id']} evidence", e.get("evidence"), "transcript", "only mentioned, not on screen")
                e["evidence"] = "transcript"


def visible(e):
    return e.get("type") == "person" and e.get("evidence") != "transcript"


# ===================================================================== names
def tok_sim(a, b):
    a, b = a.lower(), b.lower()
    if a == b:
        return 1.0
    if min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)):
        return 0.9
    return ratio(a, b)


def spelling(name, ctx):
    """Best spelling of a (person) name in the on-screen text, then the archive metadata."""
    nt = name.split()
    k, f = len(nt), ctx.c["names"]["fuzzy"]
    for label, lines in (("on_screen_text", ctx.ocr_lines), ("archive_metadata", ctx.meta_lines)):
        best = None
        for ln in lines:
            ws = re.findall(r"[A-Za-z][A-Za-z'\-]+", ln)
            for i in range(len(ws) - k + 1):
                seq = ws[i:i + k]
                if not all(w[0].isupper() for w in seq):
                    continue
                sc = [tok_sim(a, b) for a, b in zip(nt, seq)]
                ok = min(sc) >= (0.99 if k == 1 else f) and max(sc) >= 0.9
                if ok and (best is None or sum(sc) > best[0]):
                    best = (sum(sc), " ".join(w.title() if w.isupper() else w for w in seq))
        if best:
            return best[1], label
    return name, None


def gender(desc):
    w = set(re.findall(r"[a-z]+", (desc or "").lower()))
    m, f = bool(w & MALE), bool(w & FEMALE)
    return "m" if m and not f else ("f" if f and not m else None)


def fix_names(doc, ctx):
    ments = [(m, e) for m in doc["moments"] for e in persons(m) if e.get("name")]
    # 1. spelling from on-screen text / archive metadata
    for m, e in ments:
        new, src = spelling(e["name"], ctx)
        e["name_spelling_source"] = src or "model"
        if new != e["name"]:
            ctx.change(m, "names", f"{e['id']} name", e["name"], new, f"spelled as in {src}")
            e["name"] = new
    # 2. variants of one name within the video ("Bill Deon" / "Bill Deo"; "Clay" -> "Clay Pedigo")
    cnt = Counter(e["name"] for _, e in ments)
    srcs = defaultdict(set)
    for _, e in ments:
        srcs[e["name"]].add(e["name_spelling_source"])
    multi = sorted((n for n in cnt if len(n.split()) >= 2), key=lambda n: (-("model" not in srcs[n]), -cnt[n], -len(n), n))
    parent = {n: n for n in multi}

    def root(n):
        while parent[n] != n:
            n = parent[n]
        return n
    for i, a in enumerate(multi):
        for b in multi[i + 1:]:
            ta, tb = a.lower().split(), b.lower().split()
            if tok_sim(ta[0], tb[0]) >= 0.85 and ratio(ta[-1], tb[-1]) >= 0.8 and root(a) != root(b):
                parent[root(b)] = root(a)  # earlier in `multi` = better source / more frequent wins
    canon = {n: root(n) for n in multi}
    for n in cnt:
        if len(n.split()) == 1:
            hits = {canon[x] for x in multi if any(tok_sim(n, t) >= 0.9 and len(t) >= 3 for t in (x.split()[0], x.split()[-1]))}
            if len(hits) == 1:
                canon[n] = hits.pop()
    gmaj = {}
    for _, e in ments:
        g = gender(e.get("description"))
        if g:
            gmaj.setdefault(canon.get(e["name"], e["name"]), Counter())[g] += 1
    for m, e in ments:
        to = canon.get(e["name"], e["name"])
        if to == e["name"]:
            continue
        g, gm = gender(e.get("description")), gmaj.get(to)
        if g and gm and gm.most_common(1)[0][0] != g and gm[g] < gm.most_common(1)[0][1]:
            continue  # the variant describes a different-gender person: not merged
        ctx.change(m, "names", f"{e['id']} name", e["name"], to, "variant of a name used elsewhere in the video")
        e["name"] = to


# ===================================================================== appearance
def sig(desc):
    ws = [NORMALISE_WORD.get(w, w) for w in re.findall(r"[a-z][a-z\-]+", (desc or "").lower())]
    return {w for w in ws if w in WEIGHTS or w in NORMALISE_WORD.values()}


def wgt(w):
    return WEIGHTS.get(w, 0.5)


class Profiles:
    def __init__(self, ments, min_mentions):
        self.by = defaultdict(list)
        for k, (m, e) in enumerate(ments):
            self.by[e["name"]].append((k, sig(e.get("description"))))
        self.min = min_mentions

    def get(self, name, exclude=None, need=2):
        sigs = [s for k, s in self.by.get(name, []) if k != exclude]
        if len(sigs) < need:
            return None
        c = Counter(w for s in sigs for w in s)
        return {w: wgt(w) for w, n in c.items() if n / len(sigs) >= 0.5}

    def full(self):
        return {n: self.get(n, need=self.min) for n in self.by if len(self.by[n]) >= self.min}


def match(s, prof):
    return sum(prof[w] for w in s if prof and w in prof)


def fix_appearance(doc, ctx):
    c = ctx.c["names"]
    ments = [(m, e) for m in doc["moments"] for e in persons(m) if e.get("name") and visible(e)]
    P = Profiles(ments, c["min_profile_mentions"])
    full = P.full()
    done = set()

    def looks(x, name):
        p = full.get(name)
        return match(sig(x.get("description")), p) if p else 0.0
    # two named people in one moment with each other's look: swap the names
    for m in doc["moments"]:
        ps = [x for x in persons(m) if x.get("name") and visible(x)]
        for i, e1 in enumerate(ps):
            for e2 in ps[i + 1:]:
                a, b = e1["name"], e2["name"]
                if a == b or a not in full or b not in full or id(e1) in done or id(e2) in done:
                    continue
                if looks(e1, b) >= max(c["conflict_score"], looks(e1, a) + c["conflict_margin"]) and \
                        looks(e2, a) >= max(c["conflict_score"], looks(e2, b) + c["conflict_margin"]):
                    for x, nm in ((e1, b), (e2, a)):
                        x["name"] = nm
                    e1["name_source"], e2["name_source"] = e2.get("name_source"), e1.get("name_source")
                    done |= {id(e1), id(e2)}
                    ctx.change(m, "appearance", f"names swapped between {e1['id']} and {e2['id']}", a, b,
                               "each description matches the other name's appearance in the rest of the video")
                    ctx.flag(m, f"names {a!r} and {b!r} swapped by appearance; check the description")
    for k, (m, e) in enumerate(ments):
        a, s = e["name"], sig(e.get("description"))
        if not s or id(e) in done:
            continue
        own_p = P.get(a, exclude=k)
        own = match(s, own_p) if own_p else None
        scores = [(match(s, p), b) for b, p in full.items() if b != a]
        if not scores:
            continue
        bs, b = max(scores)
        if bs < c["conflict_score"] or (own is not None and bs < own + c["conflict_margin"]):
            continue
        prof_a = full.get(a)
        alt = None
        if prof_a:
            cands = [(match(sig(x.get("description")), prof_a), x["id"], x) for x in persons(m)
                     if x is not e and not x.get("name") and visible(x)]
            cands = [t for t in cands if t[0] >= c["conflict_score"]]
            alt = max(cands, key=lambda t: (t[0], t[1]))[2] if cands else None
        if alt is not None:
            alt["name"], alt["name_source"] = a, e.get("name_source")
            alt["name_spelling_source"] = e.get("name_spelling_source")
            ctx.change(m, "appearance", f"name {a!r} moved {e['id']} -> {alt['id']}",
                       why=f"{e['id']} looks like {b!r} ({e.get('description')!r}); {alt['id']} looks like {a!r}")
            ctx.flag(m, f"name {a!r} moved to another person by appearance; check the description")
        else:
            ctx.change(m, "appearance", f"{e['id']} name removed", a, None,
                       f"description {e.get('description')!r} matches {b!r}, not {a!r}")
            ctx.flag(m, f"name {a!r} removed: the described person looks like {b!r}")
        e["name"] = e["name_source"] = None
        e.pop("name_spelling_source", None)
    # man / woman conflicts within one name
    g = defaultdict(list)
    for m in doc["moments"]:
        for e in persons(m):
            if e.get("name") and gender(e.get("description")):
                g[e["name"]].append((m, e, gender(e.get("description"))))
    for name, xs in g.items():
        cnt = Counter(x[2] for x in xs)
        if len(cnt) < 2:
            continue
        (top, n1), (_, n2) = cnt.most_common(2)
        for m, e, gg in xs:
            if n1 >= 2 and n1 >= 2 * n2:
                if gg != top:
                    ctx.change(m, "appearance", f"{e['id']} name removed", name, None,
                               f"{name!r} is a {'man' if top == 'm' else 'woman'} in {n1} of {n1 + n2} mentions")
                    ctx.flag(m, f"name {name!r} removed: gender differs from its other mentions")
                    e["name"] = e["name_source"] = None
            else:
                ctx.flag(m, f"name {name!r} is given to both a man and a woman in this video")
    # descriptions are only flagged, so a two-mention look is enough evidence here
    _check_descriptions(doc, ctx, Profiles([(m, e) for m in doc["moments"] for e in persons(m) if e.get("name") and visible(e)],
                                           max(2, c["min_profile_mentions"] - 1)).full())


def _check_descriptions(doc, ctx, full):
    """Flag (never edit) description sentences that attach a name to another named person's look:
    '... the woman in orange as Zoe Royer' / 'Zoe Royer, a woman ... wearing an orange top'."""
    c = ctx.c["names"]
    for m in doc["moments"]:
        d = m.get("description") or ""
        for name, prof in full.items():
            for hit in re.finditer(re.escape(name), d):
                before = re.findall(r"[\w'-]+", d[max(0, hit.start() - 80):hit.start()])
                after_txt = d[hit.end():hit.end() + 120]
                win = []
                if before and before[-1].lower() in ("as", "named", "called"):
                    win += before[-6:]
                if re.match(r"\s*(,|in\b|wearing\b|with\b|who\b|dressed\b)", after_txt):
                    clause = re.split(r"[.;]|,\s+(?=[a-z]+s\b)", after_txt.lstrip(" ,"), maxsplit=1)[0]
                    win += re.findall(r"[\w'-]+", clause)[:12]
                s = sig(" ".join(win))
                if not s:
                    continue
                own = match(s, prof)
                others = [(match(s, p), b) for b, p in full.items() if b != name]
                if others:
                    bs, b = max(others)
                    if bs >= 1.0 and bs >= own + c["conflict_margin"]:
                        ctx.flag(m, f"description may attach {name!r} to {b!r}'s appearance")


def name_bindings(doc, ctx):
    ments = [(m, e) for m in doc["moments"] for e in persons(m) if e.get("name") and visible(e)]
    P = Profiles(ments, ctx.c["names"]["min_profile_mentions"])
    for k, (m, e) in enumerate(ments):
        p = P.get(e["name"], exclude=k, need=ctx.c["names"]["min_profile_mentions"] - 1)
        if e.get("name_source") == "on_screen_text":
            e["name_binding"] = "on_screen_text"
        elif p and match(sig(e.get("description")), p) >= 1.0:
            e["name_binding"] = "appearance_consistent"
        else:
            e["name_binding"] = "unverified"
    for m in doc["moments"]:
        for e in m.get("entities") or []:
            if e.get("name") and e.get("type") == "person" and not visible(e):
                e["name_binding"] = "not_visible"


# ===================================================================== vocabulary
REL_DROP_RX = re.compile(r"^relationship (\S+) (.+?) (\S+) dropped: predicate")


def fix_vocab(doc, ctx):
    from ..level2.metadata_generator import PREDICATES
    for m in doc["moments"]:
        ids = {e["id"] for e in m.get("entities") or []}
        rels = m.setdefault("relationships", [])
        have = {(r["subject"], r["predicate"], r["object"]) for r in rels}
        for rep in (m["annotation"].get("normalisation_repairs") or []):
            x = REL_DROP_RX.match(rep)
            if not x:
                continue
            s, p, o = x.group(1), re.sub(r"[\s\-]+", "_", x.group(2).strip().lower()), x.group(3)
            to = PREDICATE_MAP.get(p)
            if to in PREDICATES and s in ids and o in ids and s != o and (s, to, o) not in have:
                rels.append({"subject": s, "predicate": to, "object": o, "evidence": None, "restored_from": p})
                have.add((s, to, o))
                ctx.change(m, "vocab", "relationship restored", f"{s} {p} {o}", f"{s} {to} {o}", "synonym of an allowed predicate")
        acts = m.get("actions") or []
        seen = set()
        keep = []
        for a in acts:
            t = a.get("type")
            nt = ACTION_MAP.get(t, t)
            if t == "give_interview" and re.search(r"\binterviews\b", a.get("description") or ""):
                nt = "conduct_interview"  # "person_2 interviews person_1" is the interviewer's action
            if nt != t:
                ctx.change(m, "vocab", f"action {a.get('id')} type", t, nt, "synonym" if t != "give_interview" else "description says the subject interviews")
                a["type"] = nt
            key = (a["type"], tuple(a.get("participants") or []))
            if key in seen:
                ctx.change(m, "vocab", f"action {a.get('id')} removed", why=f"duplicate of another {a['type']} action")
                continue
            seen.add(key)
            keep.append(a)
        m["actions"] = keep
        if m.get("attributes") and m["attributes"].get("action") in ACTION_MAP:
            m["attributes"]["action"] = ACTION_MAP[m["attributes"]["action"]]


# ===================================================================== tags
def fix_tags(doc, ctx):
    for m in doc["moments"]:
        old = m.get("tags") or []
        new = [t for t in old if not set(re.findall(r"[a-z][a-z\-]*|\d+", t.lower())) <= TAG_DROP_WORDS]
        multi = [t for t in new if " " in t]
        new = [t for t in new if " " in t or not any(t in x.split() for x in multi)]
        if new != old:
            ctx.change(m, "tags", "tags", why=f"removed {[t for t in old if t not in new]}")
            m["tags"] = new


# ===================================================================== shot scale
def fix_shot(doc, ctx):
    from ..level2.metadata_generator import SHOT_SCALES
    c = ctx.c["shot"]
    for m in doc["moments"]:
        at, ob = m.get("attributes") or {}, m.get("observed") or {}
        v = at.get("shot_scale")
        if v is None:
            continue
        at["shot_scale_source"] = "model"
        det = {d["label"]: d["share"] for d in ob.get("shot_scale_detector") or []}
        person = 1 - det.get("no_person", 0)
        if person < c["min_person_share"] or is_screen(m) or (ob.get("people_count_median") or 0) > c["max_people"]:
            continue
        shares = {k: s / person for k, s in det.items() if k in SHOT_SCALES}
        if not shares:
            continue
        dom = max(shares, key=lambda k: (shares[k], k))
        if dom != v and shares[dom] >= c["min_dominant_share"] and shares.get(v, 0) <= c["max_model_share"]:
            at["shot_scale"], at["shot_scale_source"] = dom, "detector"
            ctx.change(m, "shot", "attributes.shot_scale", v, dom, f"pose detector {shares[dom]:.0%} of person frames")


# ===================================================================== speakers
def fix_speakers(doc, ctx):
    if not ctx.runs:
        return
    c = ctx.c["speakers"]
    for m in doc["moments"]:
        ob = m.get("observed") or {}
        if not ob.get("speakers"):
            continue
        a, b = m["start_s"], m["end_s"]
        talk = Counter()
        for r in ctx.runs:
            talk[r["speaker"]] += max(0.0, min(b, r["end"]) - max(a, r["start"]))
        tot = sum(talk.values())
        keep = [s for s in ob["speakers"] if talk[s] >= max(c["min_s"], c["min_share"] * tot)]
        if keep != ob["speakers"]:
            ctx.change(m, "speakers", "observed.speakers", ob["speakers"], keep,
                       f"speakers under {c['min_s']:g} s / {c['min_share']:.0%} of the talk time removed")
            ob["speakers"] = keep


# ===================================================================== review
def _crowd(m, n_min):
    n = 0
    for e in m.get("entities") or []:
        if e.get("type") == "person":
            n += 1
        elif e.get("type") == "person_group":
            x = re.match(r"\s*(\d+)", str(e.get("count") or ""))
            n += int(x.group(1)) if x else 2
    return n >= n_min


def recompute_review(doc, ctx):
    c = ctx.c["review"]
    for m in doc["moments"]:
        an = m["annotation"]
        keep, supp = [], []
        for r in an.get("review_reasons") or []:
            if r.startswith("relationship") and "dropped" in r:
                supp.append(f"{r} [vocabulary repair, not a content error]")
            elif r.startswith("people: metadata describes") and (_crowd(m, c["crowd_min_people"]) or is_screen(m)):
                supp.append(f"{r} [{'crowd' if _crowd(m, c['crowd_min_people']) else 'screen/video-call content'}: detector cannot count]")
            elif r.startswith("shot scale:") and ((m.get("attributes") or {}).get("shot_scale_source") == "detector" or is_screen(m)):
                supp.append(f"{r} [{'resolved: detector value used' if not is_screen(m) else 'screen content: framing detector not applicable'}]")
            else:
                keep.append(r)
        keep += m.pop("_review_add", [])
        an["review_reasons"] = list(dict.fromkeys(keep))
        an["review_reasons_suppressed"] = supp
        an["needs_review"] = bool(an["review_reasons"])


def mark_segments(doc, ctx):
    for s in doc["segments"]:
        s["single_moment"] = len(s.get("children") or []) == 1


def set_location(doc, ctx):
    loc = geo.location(ctx.src)
    doc["video"]["location"] = loc
    if loc:
        ctx.change(None, "location", "video.location", None, {k: loc[k] for k in ("country", "region", "place")}, loc["source"])


RULES = [("merge", merge_duplicates), ("ocr", clean_ocr), ("entities", fix_entities), ("names", fix_names),
         ("appearance", fix_appearance), ("vocab", fix_vocab), ("tags", fix_tags), ("shot", fix_shot),
         ("speakers", fix_speakers), ("location", set_location), ("segments", mark_segments),
         ("bindings", name_bindings), ("review", recompute_review)]


def apply_all(doc, ctx):
    for _name, fn in RULES:
        fn(doc, ctx)
    return doc
