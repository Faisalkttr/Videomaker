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
2. **Text → Stock footage video (free)**: each script line is matched to a free stock clip, with voice-over and captions. Write scenes as `what the voice says | english search keywords`. Default source is **Pixabay**: sign up at pixabay.com, then copy your key from pixabay.com/api/docs, and add it in the app or as the secret `PIXABAY_API_KEY`. Pexels is still selectable, but Pexels has paused new API keys. Contributor names are added to the description automatically.
3. **Text → AI video**: Hugging Face hosted model. Paid: Inference Providers need prepaid credits or PRO (a 402 error means no credit).
4. **Video → Video**: trim, speed, B&W, mirror, caption, fades; any source is converted to 16:9 or 9:16 (blurred background or crop).

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
