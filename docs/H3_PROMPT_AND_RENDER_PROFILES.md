# H3 prompt and render profiles

## Why there are two render phases

Prompt, identity, blocking and camera errors should be discovered on a cheap
sample, not after every shot has been rendered at formal settings. The product
therefore estimates work as:

```text
megapixels × seconds × diffusion steps
```

This is a relative compute proxy, not a runtime promise. On the locked profiles,
proof compute is about one sixth of formal compute.

| Profile | Frames | Duration | MP | Turbo | Ref size | Deliverable |
|---|---:|---:|---:|---:|---|---|
| Proof | 124 | 5.1667s | 0.4 | 6 | `match` | No |
| Production | 243 | 10.125s | 0.9 | 8 | `max` | Only after promotion |

ComfyUI's official documentation says lower megapixels run faster, the native
canvas uses a 768px short edge, the timeline snaps to `17k+5` at 24fps, and
`ref_image_size=max` can improve identity fidelity at a significant speed cost.
The selected proof values are a product experiment within the native node's
accepted ranges, not an upstream claim of universal optimum.

## Prompt shape

The compiler emits MiniMax's published field order. It preserves exact dialogue,
uses stable speaker IDs, assigns every reference one role, constrains cast count,
keeps one dominant action/camera path and excludes model-rendered delivery
subtitles. Canonical episode character IDs are compiled to the exact
`<Subject N>` associated with their approved `<Picture N>` reference, including
action states and dialogue speakers. It persists the UTF-8 prompt hash and
ordered reference-bundle hash.

The source lock and clean-room skill live under
`skills/minimax-h3-drama-director/`.

## Positive-only frame constraints

H3 Reference-to-Video consumes one positive multimodal prompt; it does not
receive the separate negative-conditioning lane used by SD image workflows.
The runtime compiler therefore must not paste a list of forbidden visual
objects into that positive prompt. In real proof review, phrases naming an
exit sign, green panel, pictogram, ceiling and door header caused those exact
concepts to remain prominent even when each phrase was preceded by `no` or
`crop out`.

Runtime contract `h3-runtime/v7-positive-only-frame-authority` expresses the
approved result instead: the frame spans subject head level to the checkout
counter; only the people, plain glass and counter are visible; interior
surfaces are uniformly dry, blank and unlettered. The graph snapshot stores the
contract version and exact prompt so a reviewer can distinguish a compiler
change from a random seed retry. SD character and scene generation still use
their real negative-conditioning inputs; this rule applies specifically to the
single positive H3 Ref2VA prompt.

## Scene plate validation

Scene plates are approved before H3 video work starts. If the bundled SD1.5
line ControlNet and Anything V5 checkpoint are available, convenience stores use
a deterministic entrance/counter/shelf layout guide; this is the preferred path
after repeated text or geography rejection because the prompt no longer has to
invent the floor plan. Otherwise, when `RealVisXL_V5.0_fp16.safetensors` is
installed, the generator uses a RealVisXL structural pass followed by a
0.62-denoise Animagine style pass. Installations missing both optional paths fall
back to single-pass Animagine. Custom scene generators keep their original
callable contract.

## Speed versus quality

- Proof uses `match` and six Turbo steps to detect concept failures cheaply.
- Formal uses `max` and eight Turbo steps because four-step generation can trade
  away motion/audio quality and fast large motion is especially fragile.
- Sage Attention remains optional and capability-detected. It may accelerate
  attention substantially, but unsupported dtypes can fall back to standard
  attention; acceleration is never treated as a quality pass.
- A technically successful MP4 cannot be delivered until content QA and human
  approval pass.
