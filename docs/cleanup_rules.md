# Cleanup stage (`clean-v2`): rule-based fixes before human review

`ingest -> level1 -> correction -> level2 -> cleanup -> review`

Deterministic and model-free; seconds per video. Reads the Level-2 document plus Level-1 OCR / diarisation /
observations and the archive metadata; never modifies them. Output: `outputs/<id>/cleanup/clean_metadata.json`
(schema `vmp-2.2`), `cleanup_log.jsonl` (one line per change: rule, moment, what, from, to, why),
`cleanup_summary.json`, `cleanup_report.md`, and the dataset copy `metadata/<video>_metadata.json/.md`.
Every change is also listed on the moment (`annotation.cleanup_changes`). Rules run in this order:

| Rule | What it does | Never does |
|---|---|---|
| merge | merges neighbouring moments of one segment with near-identical description (>= 0.75), tags (Jaccard >= 0.5), shot scale, camera motion and names, only when neither has speech (speech share <= 0.3); max 180 s; observed block recomputed; `merged_from` lists the ids | merge moments with speech (they differ by what is said) |
| ocr | `SoltanilParvazTV` -> `Soltani \| ParvazTV`, `Societyl KTV` -> `Society \| KTV`, `MOREL` -> `MORE`, `2 0 2 6` -> `2026`, `10 : 19 :12` -> `10:19:12`; a single unknown word that resembles a known or other OCR word is dropped (`Royerie`, `Farks`, `EATI`); timers keep one reading per moment; spelling corrected only for tokens with a symbol inside (`De€ember`) | "correct" real words (no dictionary): `Please`, `COMING` stay |
| entities | person described as a location -> `place`; name with company/organisation words -> `organization`; "mentioned in conversation", caller, off-screen -> `evidence: transcript` | invent entities |
| names | spelling from on-screen text, then archive title/description (`Sherry Sultani`, `Sher Sultani` -> `Sherry Soltani`); variants merged within the video (`Bill Deo` -> `Bill Deon`, `Clay` -> `Clay Pedigo` when unique); `name_spelling_source` | merge a variant whose description has the other gender |
| appearance | per name, the look (colours, accessories, garments) shared by most of its mentions; two named people with each other's look are swapped; a name on someone with another named person's look is moved to the matching unnamed person in the moment, else removed; a name given to both a man and a woman: minority removed if >= 2:1, else flagged; description sentences that attach a name to another person's look are flagged | rewrite descriptions (flag only) |
| shared | Level 2 removes a name the model gave to several people in one moment; if the name has a consistent look elsewhere in the video and exactly one of those people matches it, the name goes back to that person (`name_source` re-checked against the moment's transcript / on-screen text) | guess between two matching people |
| bindings | `name_binding` on every named person: `on_screen_text` (name shown in the moment), `appearance_consistent` (matches the name's look elsewhere), `unverified`, `not_visible` | |
| vocab | relationships dropped by Level 2 for a non-list predicate restored when it is a synonym (`conducts` -> `performs_with`, `looks_at` -> `watches`, `speaks_to` -> `talks_to`; co-presence such as `sits_with` stays dropped); action synonyms merged (`talk` -> `speak`, `play_instrument` -> `play_music`); `give_interview` described as "X interviews Y" -> `conduct_interview` | extend the closed predicate list |
| tags | removes tags made only of appearance / generic people words (`woman`, `pink blazer`, `blonde hair`, `seated`) and single words already inside another tag (`board` with `board meeting`) | |
| shot | `shot_scale` from the pose detector when it saw a person in >= 50 % of frames, one framing in >= 50 % of them (majority), the model's value in <= 20 %, <= 12 people, and the moment is not screen content (video call, slides, graphics); `shot_scale_source` = model / detector | |
| speakers | `observed.speakers` keeps speakers with >= 3 s and >= 10 % of the talk time in the moment | |
| location | `video.location {country, region, place, source}` from the archive subject tags (station location; video level only) | set moment-level `context.location` |
| segments | `single_moment: true` when a segment has one moment (its description is the moment's) | |
| review | suppresses false alarms into `review_reasons_suppressed` with the reason: dropped-predicate repairs; people counts for crowds (>= 8 described), screen content, or moments with shot cuts where the people fit into the shots (described <= detector max x (cuts + 1)); shot-scale conflicts resolved by the detector, on screen content, or where the detector itself is inconclusive (person in < 50 % of frames, or no framing in >= 50 % of person frames); a name removed from several people becomes one reason per name; adds the cleanup flags above | hide anything silently |

Thresholds: `cleanup:` in `configs/default.yaml`; tables: `src/video_metadata/cleanup/vocab.py`. Changing either is a
new cleanup version; rerun with `python -m video_metadata.cli run --stages cleanup` (or `sbatch scripts/submit_cluster.sh cleanup`).

What still needs a person (flagged, not fixed): wrong sentences in descriptions, wrong settings (a video call
called a studio), who is who among unnamed or once-named people, and content the detectors cannot see.

Test videos (batch 1, 5 videos): needs-review moments 64 -> 13; 69 review reasons suppressed (54 of them in the
orchestra concert: crowd counts and dropped predicates); orchestra 46 -> 38 moments; interview: 3 spellings of the
interviewer unified, 7 name errors fixed and flagged, 21 shot scales taken from the detector.

## clean-v2 (batch 1, 48 videos, 1,624 moments)
v1: needs review 556 -> 212. v2 adds: shared names restored by appearance, people-count check aware of shot cuts,
shot-scale override at detector majority (<= 12 people) and suppression when the detector is inconclusive.
Estimated on the v1 output: 212 -> 100 moments needing review (52 cleanup name fixes, 33 names given to several
people, 15 people counts, 1 failed generation). Shot scale now comes from the detector in ~600 of 1,624 moments:
validate the pose heuristic against the human labels on a sample before reporting camera results.
