# Moment metadata schema (outputs/<id>/level2/semantic_metadata.json -> moments[])

| Field | Content |
|---|---|
| moment_id, video_id | `<video_id>_m<NNN>` |
| start_s, end_s | seconds; always existing candidate boundaries (Level 2 never invents timestamps) |
| source_intervals | correction-stage interval ids merged into this moment |
| boundary_explanations.start/end | decision (keep / uncertain / video start/end), who decided (vlm, reconciliation, human), the VLM's evidence-based explanation, and the automatic detector evidence |
| description | 1-3 factual sentences |
| tags | searchable keywords |
| entities | `{id, type, name (null unless shown/stated), description, evidence}`; anonymous ids (person_1, group_1) |
| events | `{type, description, participants (entity ids), evidence}` |
| relationships | `{subject, predicate, object, evidence}` - only when supported, never from co-occurrence |
| context | `{setting, topic, location, notes}`; null/unknown when not supported |
| attributes | `{action, time_of_day, cinematography}` (attributes, not an organising taxonomy) |
| evidence_provenance | frame timestamps shown to the VLM, transcript track, Level-1 intervals, whether video-level context was given |
| generation_status | ok, or invalid_output (fields left empty, errors recorded; nothing is filled in) |
| review_status, needs_review | `automatic` until reviewed; needs_review when a boundary was uncertain or windows disagreed |
| pipeline_version, model, prompt_version | provenance |

`evidence` is `visual`, `transcript` or `both`. Source metadata (title, tags, description) is given to the model as
video-level context only and is never treated as proof of what a moment shows.
