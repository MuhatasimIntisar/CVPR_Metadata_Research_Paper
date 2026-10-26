# Metadata schema v2.1 (`vmp-2.1`) - outputs/<id>/level2/semantic_metadata.json

Adopted from the Chocolate Ballerina schema review (25 Oct 2026); v2.1 adds closed camera fields, detector shot
scale / camera motion, stricter name rules and review reasons. One JSON document per video:

```
{schema_version, video, annotation_set, segments[], moments[], entity_registry[]}
```

## video (source metadata - context only, never moment-level truth)
`video_id`; `source {provider, url, archive_identifier, title, description, creator, publication_date, languages,
source_tags, original_platform, role}`; `media {duration_s, fps, width, height}`.

## annotation_set
`pipeline_version, model, merge_prompt_version, metadata_prompt_version, generated_at, status` (automatic, not gold).

## segments[] (VLM-consolidated events; top level)
`segment_id` (`<video>_sNNN`), `start_s`, `end_s` (half-open, contiguous, inside the video), `description`,
`children` (moment ids), `source_intervals`, `merged_boundaries`, `boundary {start, end}` (decision, decided_by,
explanation, automatic evidence), `long_unsplit`, `needs_review`, `annotation`.

## moments[] (retrievable units)
A segment longer than `level2.max_moment_s` (180 s) is split into child moments at its candidate-interval
boundaries (pieces < 20 s joined to a neighbour); otherwise the segment has one moment with the same span.

| Field | Content / rules |
|---|---|
| moment_id, segment_id | `<video>_sNNN_mKK`; ids are opaque |
| start_s, end_s, source_intervals, boundary | span inside its segment; reasons for both edges |
| description | 1-3 factual sentences about what happens (not about editing) |
| tags | <= 12 lowercase natural-language keywords, de-duplicated |
| entities | `{id, type, name, name_source, count, description, evidence, video_entity_id?}`; type in person, person_group, object, organization, place, animal, text; `name` only if it is a proper name (has a capital letter/digit) and on one entity per moment (else removed from all and flagged); `name_source` is checked against the evidence: `transcript` / `on_screen_text` = said/shown in this span, `elsewhere_in_video` = said/shown at another time of the video (identity carried over, unverified); a name found only in the title/description is removed; interchangeable people = one person_group with `count`; ids local to the moment |
| actions | `{id, type (snake_case verb_noun, alias-normalised), description, participants (entity ids), evidence}` (renamed from `events` to avoid clashing with segments) |
| relationships | `{subject, predicate, object, evidence}`; subject and object are different entity ids; predicate from a closed list; subject acts on object (the host `interviews` the guest) (interviews, talks_to, listens_to, performs_for, performs_with, dances_with, sings_with, plays_music_with, demonstrates_to, teaches, instructs, presents, introduces, watches, applauds, holds, uses, gives_to, hugs, points_at); anything else is dropped and logged |
| context | `{setting (visible), topic (from this span's transcript), location (only if shown/said in this span), notes}` |
| attributes | `{action, shot_scale, camera_motion, camera_angle, cinematography}`; shot_scale in extreme_close_up, close_up, medium, medium_wide, wide, extreme_wide; camera_motion in static, pan, tilt, zoom, tracking, handheld; camera_angle in eye_level, high, low, overhead; null if unclear; cinematography = optional free-text note (time_of_day removed) |
| observed | detector-measured, no VLM: people count median/max, speech/music share, speakers, setting and activity shares, detector sub-events, on-screen text (channel branding and OCR misreads removed), shot cuts, `shot_scale_detector` (share of frames per framing of the largest person, from YOLO-pose keypoint visibility, plus no_person; heuristic), `camera_motion_detector` (optical flow: static, pan, zoom, handheld) |
| annotation | status (automatic), review_status (unreviewed), needs_review, review_reasons (deterministic: uncertain boundary, moment < 5 s, one name given to several people, relationship dropped, more people described than detected, shot scale contradicting the detector), generation_status, generation_errors, normalisation_repairs, attempts, model, metadata_prompt_version, evidence_provenance (frames shown, transcript track, Level-1 intervals, video context used) |

Unknown values are `null` (never "unknown"). Internal time spans are not requested from the VLM; sub-event
timing comes from the detectors (`observed.detector_sub_events`).

## entity_registry[]
Named entities linked across moments by normalised name: `{video_entity_id, name, type, name_sources, mentions,
link_method}` - unverified until review.

## Deterministic normalisation (form only, logged in annotation.normalisation_repairs)
null spellings -> null; tags lowercased and de-duplicated; action-type aliases (e.g. performing_ballet ->
perform_ballet); names without a valid source, non-proper names and names given to several entities removed;
relationships breaking the rules dropped; the model's own camera words mapped onto the closed vocabularies. No
content is added.

## Segmentation rule added in v2.1 (correction stage)
Candidate intervals shorter than `correction.min_interval_s` (5 s) are absorbed into the neighbour whose adjacent 6 s
is more similar (X-CLIP activity scores + CLIP setting); reviewer boundaries are never removed (logged as
`min_interval` in correction_log.jsonl).
