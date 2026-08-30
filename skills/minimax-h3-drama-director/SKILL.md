---
name: minimax-h3-drama-director
description: Compile an approved episodic short-drama contract into bounded, reference-aware MiniMax H3 prompts and enforce the proof-to-production quality gate.
version: 1.2.0
---

# MiniMax H3 Drama Director

Use this skill when an AI short-drama project needs MiniMax H3 shot prompts,
character/scene reference assignment, continuity control, render-profile choice,
or proof-shot acceptance. It is a clean-room product skill derived from public
upstream contracts; it does not redistribute MiniMax's guide text.

## Required inputs

Do not compile from a free-form synopsis alone. Require an approved episode
contract containing:

- one visible action and one final observable state per shot;
- stable character IDs, exact visible cast count and approved reference images;
- scene ID, layout/lighting lock, first state, final state and screen direction;
- one dominant camera path;
- exact dialogue text, speaker ID, start/end time and delivery style;
- concrete ambient sound, action sound and optional music instrumentation.

If an input is missing, stop with the matching failure code from
`references/failure-codes.md`. Never invent a replacement character, scene,
line of dialogue, reference role or continuity state during compilation.

## Compile workflow

1. Read `references/official-h3-contract.md` completely.
2. Validate duration on H3's 24fps `17k+5` frame lattice and keep one generated
   shot within 4–15 seconds.
3. Bind references in actual ComfyUI connection order. Each character binding
   must retain its canonical episode `source_id`; compile that ID to the exact
   `<Subject N>` defined from its `<Picture N>`. Use `<Picture N>` directly for
   composition/environment references. Never leave action text referring to an
   ungrounded internal character ID when a subject reference is available.
4. For Ref2VA, emit exactly these sections in order:
   `subject_definitions`, `summary`, `retention_analysis`,
   `detailed_description`, `overall_soundscape`, `non_diegetic_music`.
5. For base modes, emit exactly:
   `integrated_multimodal_description`, `overall_soundscape`,
   `non_diegetic_music`.
6. Write structural fields in English. Preserve approved dialogue and visible
   text in their original language. Format spoken lines as
   `<d>[Language] exact approved text</d>` and use stable `(S1)`, `(S2)` IDs.
7. Describe blocking, a single physical action, camera type/amplitude/speed,
   final state and concrete sound sources. H3 Ref2VA receives one positive
   multimodal prompt, so express exclusions as the approved visible state
   (`uniformly blank surfaces`, `frame spans head level to counter`) and never
   paste forbidden object names into that positive lane. Remove adjective
   stacking, contradictory camera moves, multiple competing actions and
   invented text.
8. Hash the exact UTF-8 prompt and ordered reference bundle. Persist the skill
   version, official shape, prompt hash and reference hash in the graph snapshot.
9. Follow `references/preview-production-gate.md`; a proof render is never a
   delivery asset.

## Quality contract

- Identity is controlled by approved character references, not a name in text.
- Location continuity is controlled by an approved scene/composition reference.
- The prompt must state the exact visible cast count and keep that cast stable
  throughout the shot.
- Subtitles are deterministic post-production assets. Describe all visible
  surfaces as uniformly blank and unlettered instead of listing forbidden text,
  logo or sign concepts in H3's positive prompt. H3 may generate native
  speech/audio, but delivery typography is composited only after generation.
- No clip passes on technical completion alone. It needs decoded content QA,
  first/middle/last evidence and explicit human approval bound to artifact hash.
- A promoted production job must inherit the exact approved proof prompt hash,
  reference-bundle hash and edit-selection evidence.

## Source discipline

Read `sources.lock.json` before changing this skill. Re-check upstream commits,
licenses and hashes; update the lock and tests together. Official MiniMax guide
files are linked and hashed only. Do not vendor or paraphrase large passages.
