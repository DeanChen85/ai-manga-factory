# Architecture

The product is a stateful production system, not a single ComfyUI workflow.
Every transition is persisted and recoverable.

```text
Theme + synopsis + episode count + seconds per episode
  -> MiniMax structured series/episode contract
  -> human creative approval
  -> character + scene reference generation
  -> per-asset preview and hash-bound human approval
  -> deterministic H3 director compiler
  -> low-cost proof render (non-deliverable)
  -> decoded content QA + first/middle/last evidence + human promotion
  -> formal H3 render using locked prompt/reference hashes
  -> per-shot QA + human approval
  -> episode release approval
  -> H.264/AAC 720p platform masters + subtitle/manifest/ZIP
```

## Ownership boundaries

- `story_splitter.py` asks the LLM for a structured creative contract. The LLM
  does not directly author a Comfy graph.
- `h3_director.py` deterministically compiles approved fields into the official
  H3 base or Ref2VA section shape.
- `render_video_h3.py` stages references, freezes prompt/reference hashes and
  submits a graph.
- `task_store.py` owns durable jobs, retries, proof promotion, review evidence
  and release gates.
- `render_service.py` and `series_service.py` are the only public UI facades.
- `web_app.py` is a user console; it does not write SQLite or call ComfyUI
  directly.
- `video_delivery.py` creates platform masters only from release-approved formal
  artifacts.

## Continuity model

Character identity and wardrobe come from approved character references; scene
layout and lighting come from an approved scene/composition reference. Prompt
labels follow actual ComfyUI connection order. Every shot records exact cast
count, opening state, one dominant action, final state, screen direction and a
single camera path.

The installed Ref2VA node treats pictures as multimodal identity/style/layout
references. It does not turn a picture label into a guaranteed hard keyframe.
Current upstream ComfyUI adds `MiniMaxH3AddGuide` for arbitrary-frame hard
guides; the overnight preflight reports this as a recommended capability. The
present local ComfyUI build does not expose that node, so the product does not
mislabel semantic references as hard temporal anchors.

## Audio, speech and subtitles

H3 jointly generates native audio and video, but exact lip synchronization is
probabilistic. Approved dialogue is embedded once in the H3 prompt and persisted
as a cue sheet. Delivery subtitles are deterministic post-production; they are
not requested as text inside generated frames. High-precision speech remains a
separate ADR/TTS option.

## Resilience

- SQLite stores every job and worker reservation.
- Job input hashes prevent unchanged successful work from being regenerated.
- Artifact, prompt, reference-bundle, edit-selection and decoded-visual hashes
  bind approvals to exact bytes.
- Failed jobs can resume without deleting prior audit evidence.
- Overnight execution is single-GPU leased and fail-closed on missing nodes,
  busy queue, low disk/VRAM, unsafe temperature or retry budget.

