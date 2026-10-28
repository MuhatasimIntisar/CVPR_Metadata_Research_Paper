"""Video-level location from Internet Archive subject tags (the station's location; not where each moment was shot)."""
from __future__ import annotations

import re

US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC", "washington dc": "DC"}
CA_PROVINCES = {
    "alberta": "AB", "british columbia": "BC", "manitoba": "MB", "new brunswick": "NB",
    "newfoundland and labrador": "NL", "nova scotia": "NS", "ontario": "ON", "prince edward island": "PE",
    "quebec": "QC", "saskatchewan": "SK", "yukon": "YT", "northwest territories": "NT", "nunavut": "NU"}
# subject tags that describe the station or genre, never a place
NOT_PLACE_RX = re.compile(r"\b(tv|television|media|access|community|inc|district|school|society|peg|government|"
                          r"educational|public|channel|network|news|radio|station|college|university|student|"
                          r"club|association|foundation|church|city of|town of|county of)\b", re.I)


def _subjects(src) -> list[str]:
    s = src.get("subjects") or []
    if isinstance(s, str):
        s = [x.strip() for x in re.split(r"[;,]", s.strip("[]")) if x.strip()]
    return [str(x).strip().strip("'\"") for x in s if str(x).strip()]


def location(src: dict) -> dict | None:
    """{country, region, place, source}: region = the first subject tag that is a US state / Canadian province;
    place = the next tag if it looks like a place name (not the station, not a genre tag)."""
    subj = _subjects(src)
    creator = (src.get("creator") or "").strip().lower()
    for i, tag in enumerate(subj):
        t = " ".join(tag.lower().split())
        if t in US_STATES or t in CA_PROVINCES:
            country, region = ("US", US_STATES[t]) if t in US_STATES else ("CA", CA_PROVINCES[t])
            place = None
            if i + 1 < len(subj):
                nxt = subj[i + 1]
                if nxt.lower() != creator and not NOT_PLACE_RX.search(nxt) and len(nxt.split()) <= 4 and \
                        nxt.lower() not in US_STATES and nxt.lower() not in CA_PROVINCES:
                    place = nxt
            return {"country": country, "region": region, "place": place,
                    "source": "archive subject tags (station location; video-level, not moment-level)"}
    return None
