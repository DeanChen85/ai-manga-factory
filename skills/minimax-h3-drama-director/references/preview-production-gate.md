# Proof-to-production gate

## Proof profile

- 124 frames / 24fps = 5.1667 seconds
- 0.4 megapixels
- Turbo 6 steps
- `ref_image_size=match`
- output under `previews/`
- `delivery_eligible=false`

The proof exists to validate prompt compliance, identity/cast, location,
dominant action, camera readability, speech/audio behavior and forbidden text.
It is deliberately not a platform delivery asset.

## Promotion evidence

Promotion requires all of the following on the same artifact:

- job status succeeded;
- file hash matches the stored artifact hash;
- decoded content QA passes;
- first/middle/last evidence exists;
- edit selection is current and bound to the artifact hash;
- exact prompt hash and ordered reference-bundle hash exist;
- a human confirms the contracted action and final state are visible.

The promotion record is immutable audit history. Any changed prompt, reference,
asset, output or edit selection revokes the evidence.

## Production profile

- 243 frames / 24fps = 10.125 seconds
- 0.9 megapixels
- Turbo 8 steps
- `ref_image_size=max`
- output under `videos/`
- `delivery_eligible=true` only after promotion

Production still requires automated content QA, per-shot human approval,
episode release approval and platform delivery validation.
