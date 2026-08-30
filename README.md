# AI Manga Factory

Local end-to-end pipeline for producing AI short-drama videos with MiniMax H3,
ComfyUI and a Streamlit production console. Stateful, hash-bound, fail-closed.

This is the public release of an internal engineering workbench that has been
used to drive 5-panel short episodes through ComfyUI on RTX 3090 hardware.
The pipeline, gates and 284-test offline suite are production-grade; a full
episode has not yet passed human acceptance end-to-end. See STATUS below.

## What this is

A production system, not a single ComfyUI workflow:

```
Creative Brief
  -> Series V4 contract (exact N / exact seconds)
    -> shared character / voice / world / visual / scene bible
      -> season outline + continuity state chain
        -> Episode V3 contracts
          -> shared approved assets
            -> non-deliverable proof jobs 1..N
              -> hash-bound content QA + human promotion
                -> formal render jobs 1..N
                  -> validated episode MP4 + subtitles + delivery ZIP
```

## Quick start

Prerequisites: Python 3.11+, ComfyUI 0.33.2+, NVIDIA RTX 3090 (24 GB),
MiniMax H3 model files installed under `ComfyUI/models/` (see
[docs/COMFYUI_H3_INSTALL.md](docs/COMFYUI_H3_INSTALL.md)), FFmpeg + ffprobe.

```powershell
python -m pip install -r requirements.txt
python pipeline/comfy_preflight.py        # read-only env check; must PASS
python -m streamlit run pipeline/web_app.py --server.port 8501
```

Full setup: [docs/QUICKSTART.md](docs/QUICKSTART.md). Architecture:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Render profiles and the
proof/production gate: [docs/H3_PROMPT_AND_RENDER_PROFILES.md](docs/H3_PROMPT_AND_RENDER_PROFILES.md).

## Status (2026-08-30)

Alpha / research. The 21-day engineering work in `progress.md` and
`findings.md` is the source of truth; the README link table here points at
the in-repo documentation. A 5-panel project (`ep_h3_skill_smoke_20260824`)
is in progress: p01 / p02 succeeded at 720x1280 / 24 fps / h264 + aac 48
kHz; p03 QA-rejected on action-timing; p04 / p05 queued. No "ready for
production" claim is made here.

## Repository contents

- `pipeline/` 33 Python modules in 5 layers (UI facade -> services ->
  stores -> adapters; documented in `docs/ARCHITECTURE.md`)
- `tests/` 21 test files, 284 offline tests (run with `pytest -q`)
- `skills/minimax-h3-drama-director/` Source-locked H3 prompt compiler
  with failure codes and a proof/production gate spec
- `docs/` Public-facing documentation (architecture, quickstart, install,
  prompt/render profiles, external-project adoption)
- `examples/creative_brief.json` Sample creative brief
- `.github/workflows/ci.yml` Offline CI on Windows x Python 3.11 / 3.12
- `scripts/` is intentionally not bundled; per-environment tunnelling
  scripts (ngrok / cloudflared) belong outside version control

## What is *not* in this repository

- MiniMax H3 model weights (download separately, see install doc)
- ComfyUI runtime (install separately)
- Custom nodes (install separately; this repo only documents which ones)
- Generated media from `output/`
- SQLite task state from `state/`
- Any local credentials / API keys / machine-specific paths

All of the above are covered by `.gitignore`.

## Testing

The full suite is hermetic: no paid API, no GPU, no FFmpeg dependency.

```powershell
python -m pytest -q
```

The CI workflow at `.github/workflows/ci.yml` additionally runs
`python pipeline/release_preflight.py` (offline secret scan).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
Third-party components retain their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Security

Never commit `.env`, API keys or local credentials. See
[SECURITY.md](SECURITY.md) and [CONTRIBUTING.md](CONTRIBUTING.md).