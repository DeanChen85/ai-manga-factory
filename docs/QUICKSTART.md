# Quick start

## 1. Prerequisites

- Windows or Linux with Python 3.11+
- NVIDIA GPU and a working ComfyUI installation
- FFmpeg and ffprobe on `PATH` or configured by environment variables
- MiniMax API key for story/shot contract generation
- MiniMax H3 model files accepted and installed under their own upstream license

The application repository does not redistribute model weights.

## 2. Install the web runtime

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit only the local `.env`. At minimum set `COMFYUI_ROOT`; set
`MiniMax_API_KEY` in a process environment or secret manager. Never commit
`.env` or `minimax api.txt`.

## 3. Validate ComfyUI

Start ComfyUI, then verify:

```powershell
Invoke-RestMethod http://127.0.0.1:8188/system_stats
```

The production graph needs native H3 Ref2VA, Turbo, Sage Attention, video/audio
VAE and video-save nodes. Follow [the pinned H3 installation](COMFYUI_H3_INSTALL.md),
then run `python pipeline/comfy_preflight.py`. It reports exact missing nodes,
model filenames and media tools before any GPU job starts.

## 4. Start the product

```powershell
python -m streamlit run pipeline/web_app.py --server.port 8501
```

On Windows, `启动.bat` performs the same dependency and port checks without
hard-coded developer paths or automatic package installation.

## 5. First user journey

1. Enter theme, synopsis, style, episode count and seconds per episode.
2. Generate the structured contract and review story, characters and shots.
3. Approve the creative contract.
4. Generate character and scene references; preview and approve each asset.
5. Start video production. The default strategy creates non-deliverable proof
   clips first.
6. For every proof, inspect the video, first/middle/last frames, exact H3 prompt,
   reference roles and hashes. Promote only when the contracted action and final
   state are visible.
7. Run formal production, approve the formal shots, then approve episode release.
8. Export the 720p platform master and its manifest/subtitle/ZIP package.

## 6. Tests and release check

```powershell
python -m unittest discover -s tests -p "test_*.py"
python pipeline/release_preflight.py
```

Both commands are offline and do not submit paid MiniMax or GPU jobs.
