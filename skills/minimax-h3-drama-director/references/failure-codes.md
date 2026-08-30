# Fail-closed director codes

| Code | Meaning | Required action |
|---|---|---|
| `H3-CONTRACT-MISSING` | Required shot/identity/scene field absent | Repair the creative contract; do not compile |
| `H3-ACTION-MULTIPLE` | More than one dominant visible action | Split into separate shots |
| `H3-CAMERA-CONFLICT` | Camera paths contradict each other | Select one dominant path |
| `H3-CAST-UNBOUND` | Visible character lacks approved identity reference | Generate and approve the reference first |
| `H3-REFERENCE-ORDER` | Prompt label does not match graph connection order | Rebuild bindings and prompt together |
| `H3-DIALOGUE-MUTATED` | Generated line differs from approved text | Restore exact dialogue and speaker timing |
| `H3-TEXT-INVENTED` | Model is asked to render delivery subtitles/UI text | Remove it and use deterministic post-production |
| `H3-PROMPT-BUDGET` | Prompt is under-specified or over-complex | Rewrite within the bounded contract |
| `H3-PROOF-NOT-PROMOTED` | Formal render requested without approved proof | Complete evidence-bound promotion |
| `H3-CAPABILITY-MISSING` | Required ComfyUI node/model is absent | Stop and update/install the declared dependency |

