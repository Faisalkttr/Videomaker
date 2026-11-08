# YouTube-Ready Video Generator (Streamlit)

Every mode exports a **YouTube package** (ZIP): `video.mp4`, `thumbnail.jpg` (16:9 only), `metadata.txt`.

## Output spec
- 1920x1080 (video) or 1080x1920 (Short), 30 fps
- H.264 High profile, yuv420p, `faststart`, ~8 Mbps
- AAC audio, 48 kHz, 192 kbps
- Thumbnail 1280x720 JPEG (under 2 MB)
- Shorts get `#Shorts` appended to the description

## Modes
1. **Text → Narrated video**: script lines become scenes with AI voice-over (gTTS, many languages incl. Arabic), optional background music.
2. **Text → AI video**: Hugging Face hosted model (needs a free token; short clips, can be repeated).
3. **Video → Video**: trim, speed, B&W, mirror, caption, fades; any source is converted to 16:9 or 9:16 (blurred background or crop).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free
1. Push `app.py`, `requirements.txt`, `README.md` to GitHub.
2. share.streamlit.io → New app → select repo and `app.py`.
3. For AI mode add secret `HF_TOKEN = "hf_..."` under Advanced settings → Secrets.

## Monetisation reality check
- You must join the YouTube Partner Program (thresholds are in YouTube Studio).
- YouTube demonetises mass-produced / repetitive ("inauthentic") content. Add original scripts, commentary and editing; don't bulk-upload raw template output.
- Disclose realistic synthetic media via the "Altered content" setting.
- Use only music and footage you have rights to.
- Free Streamlit hosting has limited CPU/RAM: keep videos short (a few minutes) or run locally for longer ones.
