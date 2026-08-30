# H3 public prompt contract used by this product

This is a concise, clean-room implementation note. The authoritative sources
and immutable revisions are recorded in `../sources.lock.json`.

## Modes and section order

Base T2VA/I2VA/FL2VA/L2VA prompts use:

1. `integrated_multimodal_description`
2. `overall_soundscape`
3. `non_diegetic_music`

Ref2VA prompts use:

1. `subject_definitions`
2. `summary`
3. `retention_analysis`
4. `detailed_description`
5. `overall_soundscape`
6. `non_diegetic_music`

Section labels and order are machine contracts. Do not translate or rename
them. Structural prose is English; approved dialogue or visible text retains
its source language.

## Reference semantics

- Reference order is connection order, not filename order.
- `<Subject N>` represents a reusable subject identity defined from a connected
  `<Picture N>` reference.
- A standalone `<Picture N>` is appropriate for composition, a key visual or an
  environment authority.
- Every reference gets one explicit job. Do not ask the model to guess whether
  an image controls identity, wardrobe, layout, lighting or final composition.
- ComfyUI Ref2VA supports up to nine images, three videos and three audio clips.
  `match` is the speed-oriented image-size path; `max` retains stronger identity
  detail and may be several times slower.

## Shot writing

- First describe the opening state; add timestamps only for subsequent changes.
- One short-drama source clip gets one dominant action and one camera path.
- State camera type, amplitude and speed in observable terms.
- Give a measurable final state so QA can compare beginning and end.
- Dialogue uses stable speaker IDs and exact approved wording:
  `<d>[Chinese] 台词原文</d>`.
- Soundscape contains ambient/action/non-verbal sources, not music.
- Music names instrumentation, tempo and dynamics; use `N/A` when absent.
- Avoid abstract mood strings, adjective soup, conflicting motions, impossible
  body actions and model-rendered subtitles.
- Ref2VA has one positive multimodal prompt. Encode exclusions as affirmative
  visible states and framing boundaries; do not enumerate forbidden objects in
  that positive lane because the named concepts can leak into generated frames.

## Timing and canvas

- H3 outputs at 24fps and uses the `17k+5` frame lattice.
- 124 frames is about 5.167 seconds; 243 frames is 10.125 seconds.
- The local/native node's trained range is approximately 124–362 frames.
- Resolution must be a multiple of 32. The product uses megapixel profiles and
  lets ComfyUI calculate the exact width/height for the selected aspect ratio.
