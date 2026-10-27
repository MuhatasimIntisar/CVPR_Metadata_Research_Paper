# Corpus selection (metadata only)

Selects exactly 1,000 Internet Archive videos for the TVM dataset from the three discovery rounds, using only the
metadata already collected (titles, descriptions, IA subject tags, discovery categories and terms, station, dates,
duration, size, captions, checksums). No video, thumbnail, model or GPU is used; network access is optional
(stage 2 availability check only). Every parameter is in `selection_config.json`; paths are relative to the
repository root, so the scripts can be started from any folder.

| Stage | Script | Reads | Writes (`outputs/…`) |
|---|---|---|---|
| 1 Consolidation | `01_consolidate.py` | the 3 discovery CSVs, `collection/categories.json`, pilot manifest / `corpus/` | `01_consolidation/` candidate_master, duplicate_groups, suspected_reuploads, pilot_exclusions, report |
| 2 Eligibility | `02_eligibility.py` | candidate_master | `02_eligibility/` eligible_candidates (status + reasons + flags), report, optional availability_cache |
| 3 Profiling | `03_profile_metadata.py` | eligible_candidates, suspected_reuploads | `03_profile/` metadata_profiles, programme_families, related_video_pairs, topic_clusters, diversity report, `cache/` (TF-IDF) |
| 4 Selection | `04_select.py` | profiles, related pairs | `04_selection/` selected_1000, reserve_candidates, selection_decisions, selection_comparison, report |
| 5 Audit + freeze | `05_report_and_freeze.py` | stages 1-4 | `05_final/` human_audit, corpus_manifest_v1.{csv,json,sha256}, download/download_list_v1.csv, reserve_manifest, final_selection_report, plots |

Shared code: `common.py` (I/O, normalisation, category re-matching, title keys, geography), `metrics.py`
(set statistics). Tests: `python -m pytest collection/selection/tests -q` (synthetic data only).

## What each stage decides
1. **Consolidation** - records with the same identifier are merged (rounds, discovery categories and matched terms
   kept); identical video MD5 under different identifiers -> one kept (captions > rounds > earliest date), others
   excluded; pilot identifiers excluded; probable re-uploads (same station, title similarity >= 0.92, duration
   within 30 s / 2 %, no differing episode/date tokens; or identical normalised title on another station) are
   **flagged only**. Every candidate is re-matched against all ten categories (whole-word match on title and IA
   subject tags); the discovery query is kept as evidence too. Labels are keyword evidence, not verified content.
2. **Eligibility** - `excluded` only for a real blocker (no video file, invalid duration, outside 5-90 min, file
   > 2 GB, invalid identifier, no title, or unavailable at a live check); caveats (no captions, no description,
   upload date only, > 600 MB, suspected re-upload, ...) make a video `conditional`, never excluded.
3. **Profiling** - station = IA creator (identifier prefix only when no creator); programme families from
   episode markers / recurring title templates / shared title prefixes on one station; TF-IDF (titles + descriptions
   with station boilerplate removed) -> cosine neighbours and 40 LSA k-means topic clusters; region from IA subject
   tags (source) or the station name (inferred) - the broadcaster's location, not necessarily the filming location.
4. **Selection** - greedy marginal gain (objective in the `04_select.py` docstring): diminishing-returns coverage
   of categories, topics, stations, regions, quarters and duration bands (each family normalised by its number of
   values), captions as a multiplier, a capped bonus for 2nd-4th episodes of confidently identified series, and a
   capped bonus for supported related videos. Hard constraints: exactly N, hours budget, minimum per category,
   no possible-re-upload pair. Then up to 2 passes of local swaps. Four strategies are compared.
5. **Freeze** - stratified human-audit sample; manifest sorted by identifier with SHA-256; downloader list in the
   exact `ia_download.py` column layout with `select = x`. Status stays `provisionally_frozen` until the audit is
   declared complete and every selected video passed a live availability check.

Relations (`same_series`, `same_station_near_date`, `similar_topic`, `same_event`, `possible_reupload`) are
*potential* retrieval distractors from metadata, not verified hard negatives. `series_id` should also be used when
making train/test splits so episodes of one programme do not leak across splits.
