# Changelog

All material product changes are recorded here. Generated media, local model
weights, machine paths and secrets are never part of the changelog.

## Unreleased — 2026-08-28 acceptance hardening

### Added

- Hash-bound paired first/final composition anchors for single-character H3
  shots after a visual rejection.
- Explicit stable promotion from an approved non-deliverable proof to a 720p
  production master. The promotion does not call a model; it deterministically
  scales/pads, normalizes 24 fps and 48 kHz stereo, reruns decoded-video QA and
  still requires independent human approval.
- Stable promotion can replace an unsubmitted queued formal render when no
  episode worker or remote `prompt_id` exists, preventing an approved proof
  from being destroyed by a second random generation.
- ComfyUI resource release after a completed prompt, with queue-empty checks
  before `/free` unload/free-memory requests.
- Delivery ZIP write-after validation plus exact one-video/one-audio and
  audio/video duration-delta gates.
- External-project decision log and GitHub quality-regression issue template.

### Fixed

- H3 first/final Picture bindings now use distinct approved image hashes for
  single-character state changes.
- Stable-production decoded QA is bound to the installed `output_path`, not the
  deleted temporary FFmpeg path.
- Stable-production prompt review continues to use the immutable approved H3
  graph; the deterministic transform manifest no longer replaces it.
- Formal rerender rejection no longer forces repeated random retries when the
  previously approved proof remains cryptographically valid.
- H3 action-prompt compaction no longer leaves dangling connector fragments;
  long canonical actions retain immutable audit text/hash while the runtime
  fragment keeps the actor, verb and target motion.
- Human timing rejection feedback now produces an explicit frame-zero start,
  deadline and held-final-state correction in the next H3 prompt.
- Hand-object shots fail closed unless the current first/final composition
  anchors are approved against the current panel action/camera contract hash.
- Web task progress now distinguishes ComfyUI GPU execution from a remotely
  pending prompt and shows the real queue position when available.

### Acceptance evidence

- Real Streamlit flow: `ep_h3_skill_smoke_20260824`.
- p01 and p02 approved proof content was promoted to 720×1280, H.264, 24 fps,
  AAC 48 kHz stereo; decoded QA and independent three-part human review passed.
- Remaining p03–p05 proof/production/release acceptance stays open until their
  current artifacts pass the same gates. No completion claim is made here.
- p03 v12 first/final anchors passed human review, but its first H3 proof was
  rejected because wallet handoff completed after the 1.35-second contract
  deadline. A later ComfyUI process exit was recovered only through the Web
  infrastructure-retry gate; no missing remote prompt was blindly duplicated.
