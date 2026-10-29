# Annotation viewer

One page that plays a video next to every annotation layer: detector steps (activity, setting, people, speech, music,
speaker, shot scale, camera motion, shot cuts, on-screen text, topic change), all boundary evidence, the pooled
rule decisions, candidate intervals, Level-2 merge/keep decisions, segments and moments (full metadata, cleanup
changes, review reasons). Two runs of the same video (e.g. 4B vs 27B) can be compared lane by lane.

1. Make bundles (where the outputs are, e.g. on Kelvin from `code/`):
   `python scripts/viewer_bundle.py ../outputs --out ../viewer_bundles`
   (an experiment root works too: `python scripts/viewer_bundle.py ../outputs_ab/q27b --out ../viewer_bundles/q27b`)
2. Copy `viewer_bundles/` to the laptop and open `viewer/index.html` in Chrome or Edge (no server needed).
3. "Bundle folder…" -> the bundles folder; "Corpus folder…" -> `corpus/` (each video is matched to its bundle), or
   "Video file…" for a single video.

Click any lane or a row of the moment list to inspect it and jump the video there. Keys: space, arrows (5 s),
`[` / `]` previous / next moment, `+` / `-` zoom, Ctrl + wheel zoom.
