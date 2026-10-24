# Correction rules (configs/default.yaml -> correction)

All rules are deterministic, configurable and logged (`correction/correction_log.jsonl`); they are initial design
parameters, not tuned per video.

Step state (2 s steps), rule B: `unobservable` (black frame) | `no person visible` | top-1 X-CLIP activity when its
score >= 2x uniform and >= 1.15x the runner-up | `uncertain`. Lower-ranked labels are never promoted.

Noise floor: a state needs >= 4 s. A shorter run between two runs of the same state is bridged; otherwise it becomes
`uncertain`. Uncertain gaps between runs of the same state are bridged and never create a boundary (rule D).

Evidence and weights (pooled within 2 s; each type counted once; boundary if sum >= 1.0):

| Evidence | Weight | Condition |
|---|---:|---|
| activity | 1.0 | new state persists >= 6 s (rule A) |
| activity_short | 0.6 | persists >= 4 s; needs environment or visual support (rule A) |
| activity_family | 0.5 | change between variants of one activity (talking seated / to camera / standing) |
| observability | 1.0 | entering/leaving a black span >= 4 s |
| environment | 0.6 | CLIP environment top-1 change lasting >= 2 frames (10 s) |
| visual_transition | 0.4 | CLIP frame cosine < 0.80; a shot cut alone = 0.2; cut + CLIP jump count once (rule E) |
| people | 0.4 | person-count bin jump >= 2 between 6 s windows; 0.2 if a cut is within one step (rule C) |
| audio | 0.3 | music or speech onset/offset |

Consequences: a cut, a reframing (people + visual = 0.8) or audio alone never ends a moment; continuing speech
never blocks a new visual activity. Short certain activities (>= 4 s) inside a moment are kept as `sub_events`
(rule F); activity changes inside a moment remain evidence for Level 2 (rule G).

Human overrides (`review/overrides.json`): activity overrides, forced and removed boundaries. They are applied first
and win over every automatic rule.
