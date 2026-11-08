import io
import os
import tempfile
import textwrap
import zipfile

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    afx,
    concatenate_videoclips,
)
from moviepy.video.fx.all import blackwhite, crop, fadein, fadeout, mirror_x, speedx
from moviepy.video.fx.all import loop as vloop
import requests


def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

st.set_page_config(page_title="YouTube Video Generator", page_icon="🎬", layout="wide")
st.title("🎬 YouTube-Ready Video Generator")
st.caption("Text → video and video → video. Output: MP4 + thumbnail + metadata, ready for YouTube Studio.")

# ------------------------------------------------------------------ presets
PRESETS = {
    "YouTube video (16:9 · 1920x1080)": {"size": (1920, 1080), "short": False},
    "YouTube Short (9:16 · 1080x1920)": {"size": (1080, 1920), "short": True},
}
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSans-Bold.ttf",
]


def get_font(size):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def tmp_path(suffix=".mp4"):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.close()
    return f.name


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def wrap_draw(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = f"{cur} {w}".strip()
        if draw.textlength(test, font=font) <= max_w or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def text_card(text, size, colors, font_size):
    w, h = size
    top, bottom = np.array(colors[0], float), np.array(colors[1], float)
    ramp = np.linspace(0, 1, h)[:, None, None]
    img = Image.fromarray(((top * (1 - ramp) + bottom * ramp) * np.ones((h, w, 3))).astype("uint8"))
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)
    lines = wrap_draw(draw, text, font, w * 0.82)
    lh = font_size * 1.3
    y = (h - lh * len(lines)) / 2
    for line in lines:
        tw = draw.textlength(line, font=font)
        draw.text(((w - tw) / 2, y), line, font=font, fill="white", stroke_width=2, stroke_fill="black")
        y += lh
    return img


def overlay_caption(clip, text, font_size=56):
    w, h = clip.size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)
    lines = wrap_draw(draw, text, font, w * 0.85)
    lh = font_size * 1.25
    y = h * 0.78 - lh * len(lines) / 2
    for line in lines:
        tw = draw.textlength(line, font=font)
        x = (w - tw) / 2
        draw.rectangle([x - 14, y - 4, x + tw + 14, y + lh - 8], fill=(0, 0, 0, 170))
        draw.text((x, y), line, font=font, fill="white")
        y += lh
    cap = ImageClip(np.array(img)).set_duration(clip.duration)
    return CompositeVideoClip([clip, cap])


# ------------------------------------------------------------------ fit / audio / export
def blur_frame(f):
    img = Image.fromarray(f)
    small = img.resize((max(1, img.width // 8), max(1, img.height // 8)))
    small = small.filter(ImageFilter.GaussianBlur(3))
    return np.array(small.resize(img.size))


def fit_clip(clip, size, mode):
    """Make any clip exactly `size` (so YouTube/Shorts accepts it as 16:9 or 9:16)."""
    W, H = size
    if tuple(clip.size) == (W, H):
        return clip
    cover = max(W / clip.w, H / clip.h)
    bg = clip.resize(cover)
    bg = crop(bg, x_center=bg.w / 2, y_center=bg.h / 2, width=W, height=H)
    if mode == "Crop to fill":
        return bg
    bg = bg.without_audio().fl_image(blur_frame)
    fg = clip.resize(min(W / clip.w, H / clip.h)).set_position("center")
    return CompositeVideoClip([bg, fg], size=(W, H)).set_duration(clip.duration)


def add_music(clip, music_path, volume):
    m = AudioFileClip(music_path)
    if m.duration < clip.duration:
        m = afx.audio_loop(m, duration=clip.duration)
    else:
        m = m.subclip(0, clip.duration)
    m = afx.audio_fadeout(m.volumex(volume), 1.5)
    audio = CompositeAudioClip([clip.audio, m]) if clip.audio else m
    return clip.set_audio(audio)


def export_youtube(clip, path):
    """H.264 High / yuv420p / AAC 48 kHz / 30 fps / faststart: YouTube's recommended upload spec."""
    clip.write_videofile(
        path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        audio_fps=48000,
        audio_bitrate="192k",
        bitrate="8M",
        preset="medium",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"],
        logger=None,
    )


def make_thumbnail(clip, title, size=(1280, 720)):
    t = min(1.0, clip.duration / 2)
    frame = Image.fromarray(clip.get_frame(t)).convert("RGB")
    scale = max(size[0] / frame.width, size[1] / frame.height)
    frame = frame.resize((int(frame.width * scale), int(frame.height * scale)))
    left, top = (frame.width - size[0]) // 2, (frame.height - size[1]) // 2
    frame = frame.crop((left, top, left + size[0], top + size[1]))
    dark = Image.new("RGB", size, (0, 0, 0))
    frame = Image.blend(frame, dark, 0.25)
    draw = ImageDraw.Draw(frame)
    font = get_font(90)
    lines = wrap_draw(draw, title.upper(), font, size[0] * 0.88)[:3]
    y = (size[1] - 105 * len(lines)) / 2
    for line in lines:
        tw = draw.textlength(line, font=font)
        draw.text(((size[0] - tw) / 2, y), line, font=font, fill="white", stroke_width=6, stroke_fill="black")
        y += 105
    buf = io.BytesIO()
    frame.save(buf, "JPEG", quality=90)  # YouTube limit: 2 MB
    return buf.getvalue()


def finalize(clip):
    """Fit -> music -> export -> thumbnail -> metadata -> ZIP, then show everything."""
    cfg = PRESETS[preset]
    with st.spinner("Encoding for YouTube..."):
        clip = fit_clip(clip, cfg["size"], fit_mode)
        if music_file is not None:
            mp = tmp_path(os.path.splitext(music_file.name)[1] or ".mp3")
            with open(mp, "wb") as f:
                f.write(music_file.getvalue())
            clip = add_music(clip, mp, music_vol)
        out = tmp_path()
        export_youtube(clip, out)
        with open(out, "rb") as f:
            video_bytes = f.read()
        thumb = None if cfg["short"] else make_thumbnail(clip, title or "My Video")
        desc = description
        if cfg["short"] and "#shorts" not in desc.lower():
            desc = (desc + "\n\n#Shorts").strip()
        meta = (
            f"TITLE ({len(title)}/100):\n{title}\n\n"
            f"DESCRIPTION:\n{desc}\n\n"
            f"TAGS:\n{', '.join(t.strip() for t in tags.split(',') if t.strip())}\n\n"
            "CHECKLIST for YouTube Studio:\n"
            "- Audience: set 'Made for kids' correctly.\n"
            "- 'Altered content': answer Yes if the video has realistic synthetic people/places/events.\n"
            "- Only use music/footage you own or have a licence for.\n"
        )
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("video.mp4", video_bytes)
            z.writestr("metadata.txt", meta)
            if thumb:
                z.writestr("thumbnail.jpg", thumb)
    dur = clip.duration
    if cfg["short"] and dur > 180:
        st.warning(f"This Short is {dur:.0f}s. Shorts must be 3 minutes or less to be treated as a Short.")
    if not cfg["short"] and dur < 480:
        st.caption("Tip: videos of 8+ minutes can carry mid-roll ads once you are monetised.")
    st.success(f"Ready: {cfg['size'][0]}x{cfg['size'][1]} · 30 fps · H.264/AAC · {dur:.0f}s")
    st.video(video_bytes)
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("⬇️ video.mp4", video_bytes, "video.mp4", "video/mp4")
    if thumb:
        c2.download_button("⬇️ thumbnail.jpg", thumb, "thumbnail.jpg", "image/jpeg")
    c3.download_button("⬇️ metadata.txt", meta, "metadata.txt")
    c4.download_button("📦 Everything (.zip)", zbuf.getvalue(), "youtube_package.zip", "application/zip")
    if thumb:
        st.image(thumb, caption="Auto thumbnail (1280x720)", width=480)
    st.text_area("Metadata preview", meta, height=220)


# ------------------------------------------------------------------ sidebar: YouTube settings
st.sidebar.header("YouTube output")
preset = st.sidebar.radio("Format", list(PRESETS))
fit_mode = st.sidebar.radio("If source doesn't match", ["Blurred background", "Crop to fill"])
title = st.sidebar.text_input("Title (max 100)", "My Awesome Video", max_chars=100)
description = st.sidebar.text_area("Description", "Made with Streamlit.")
tags = st.sidebar.text_input("Tags (comma separated)", "video, tutorial")
st.sidebar.subheader("Background music (optional)")
music_file = st.sidebar.file_uploader("Music you have rights to", type=["mp3", "wav", "m4a"])
music_vol = st.sidebar.slider("Music volume", 0.0, 1.0, 0.15, 0.05)

with st.expander("⚠️ Read before monetising on YouTube"):
    st.markdown(
        """
- Monetisation requires joining the **YouTube Partner Program** (subscriber and watch-hour/Shorts-view thresholds; check the current numbers in YouTube Studio).
- YouTube demonetises **mass-produced / repetitive / "inauthentic" content**. Plain text slideshows or unedited AI clips uploaded in bulk usually won't qualify. Add your own script, voice, commentary, editing and a clear topic.
- Use **Altered content** disclosure when realistic synthetic media is shown.
- Don't use copyrighted music or footage you don't have rights to.
        """
    )

mode = st.radio(
    "Mode",
    ["Text → Narrated video", "Text → Stock footage video (free)", "Text → AI video (Hugging Face)", "Video → Video"],
    horizontal=True,
)

# ------------------------------------------------------------------ Mode 1: narrated text video
if mode.startswith("Text → Narrated"):
    st.subheader("Text → Narrated video")
    script = st.text_area(
        "Script (one scene per line)",
        "Welcome to the channel\nToday: five facts about space\nFact one: a day on Venus is longer than its year\nThanks for watching, subscribe for more!",
        height=180,
    )
    c1, c2, c3, c4 = st.columns(4)
    secs = c1.slider("Min seconds/scene", 2.0, 10.0, 3.0, 0.5)
    fsize = c2.slider("Font size", 50, 160, 90)
    narrate = c3.checkbox("AI voice-over (gTTS)", True)
    lang = c4.selectbox("Voice language", ["en", "ar", "fr", "es", "de", "hi", "tr"])
    k1, k2 = st.columns(2)
    col_a = k1.color_picker("Gradient top", "#14224a")
    col_b = k2.color_picker("Gradient bottom", "#782878")
    if st.button("Create YouTube package", type="primary"):
        lines = [l.strip() for l in script.splitlines() if l.strip()]
        if not lines:
            st.error("Write at least one line.")
        else:
            size = PRESETS[preset]["size"]
            with st.spinner("Building scenes..."):
                clips = []
                for line in lines:
                    dur, audio = secs, None
                    if narrate:
                        try:
                            from gtts import gTTS

                            ap = tmp_path(".mp3")
                            gTTS(line, lang=lang).save(ap)
                            audio = AudioFileClip(ap)
                            dur = max(secs, audio.duration + 0.6)
                        except Exception as e:
                            st.warning(f"Voice-over skipped for one line: {e}")
                    img = text_card(line, size, (hex_rgb(col_a), hex_rgb(col_b)), fsize)
                    c = fadeout(fadein(ImageClip(np.array(img)).set_duration(dur), 0.4), 0.4)
                    if audio is not None:
                        c = c.set_audio(audio)
                    clips.append(c)
                final = concatenate_videoclips(clips, method="compose")
            finalize(final)

# ------------------------------------------------------------------ Mode: stock footage (Pexels)
elif mode.startswith("Text → Stock"):
    st.subheader("Text → Stock footage video (free clips)")
    st.info(
        "Free key: sign up at pixabay.com, then open pixabay.com/api/docs to see your key. "
        "Write each scene as: `what the voice says | english search keywords`. "
        "The part after | is optional; without it the sentence itself is used as the search."
    )
    source = st.radio(
        "Footage source",
        ["Pixabay (free key)", "Pexels (only if you already have a key; new keys are paused)"],
        horizontal=True,
    )
    is_pixabay = source.startswith("Pixabay")
    pkey = st.text_input(
        f"{'Pixabay' if is_pixabay else 'Pexels'} API key",
        value=get_secret("PIXABAY_API_KEY" if is_pixabay else "PEXELS_API_KEY"),
        type="password",
    )
    script = st.text_area(
        "Script (one scene per line)",
        "Space is bigger than you think | galaxy stars night sky\n"
        "Light from the Sun takes eight minutes to reach us | sun sunrise\n"
        "Subscribe for more space facts | telescope observatory",
        height=200,
    )
    c1, c2, c3, c4 = st.columns(4)
    secs = c1.slider("Min seconds/scene", 3.0, 12.0, 5.0, 0.5)
    narrate = c2.checkbox("AI voice-over (gTTS)", True)
    lang = c3.selectbox("Voice language", ["en", "ar", "fr", "es", "de", "hi", "tr"])
    show_caps = c4.checkbox("Show captions", True)

    def stock_pick(use_pixabay, key, query, portrait, used):
        """Search the chosen stock site and download one unused clip. Returns (path, id, author) or None."""
        cands = []  # (id, author, [ {url, width, height} ])
        if use_pixabay:
            r = requests.get(
                "https://pixabay.com/api/videos/",
                params={"key": key, "q": query[:100], "per_page": 20, "safesearch": "true"},
                timeout=30,
            )
            r.raise_for_status()
            for v in r.json().get("hits", []):
                files = [
                    {"url": f["url"], "width": f["width"], "height": f.get("height") or 0}
                    for f in (v.get("videos") or {}).values()
                    if f.get("url") and f.get("width")
                ]
                if files:
                    cands.append((v["id"], v.get("user", "Pixabay contributor"), files))
        else:
            r = requests.get(
                "https://api.pexels.com/videos/search",
                headers={"Authorization": key},
                params={"query": query, "per_page": 10, "orientation": "portrait" if portrait else "landscape"},
                timeout=30,
            )
            r.raise_for_status()
            for v in r.json().get("videos", []):
                files = [
                    {"url": f["link"], "width": f["width"], "height": f["height"]}
                    for f in v.get("video_files", [])
                    if f.get("file_type") == "video/mp4" and f.get("width")
                ]
                if files:
                    cands.append((v["id"], v.get("user", {}).get("name", "Pexels contributor"), files))
        # first pass: matching orientation; second pass: anything unused (fit_clip handles the rest)
        for need_orientation in (True, False):
            for vid, author, files in cands:
                if vid in used:
                    continue
                if need_orientation and not any((f["height"] > f["width"]) == portrait for f in files):
                    continue
                pool = [f for f in files if (f["height"] > f["width"]) == portrait] if need_orientation else files
                best = min(pool, key=lambda f: abs(min(f["width"], f["height"]) - 1080))
                data = requests.get(best["url"], timeout=180).content
                path = tmp_path()
                with open(path, "wb") as f:
                    f.write(data)
                return path, vid, author
        return None

    if st.button("Create YouTube package", type="primary"):
        lines = [l.strip() for l in script.splitlines() if l.strip()]
        if not pkey:
            st.error("Please add your stock-footage API key.")
        elif not lines:
            st.error("Write at least one line.")
        else:
            size = PRESETS[preset]["size"]
            portrait = PRESETS[preset]["short"]
            used, credits, clips = set(), [], []
            try:
                prog = st.progress(0.0, text="Fetching clips...")
                for i, raw in enumerate(lines):
                    narration, _, kw = raw.partition("|")
                    narration, query = narration.strip(), (kw.strip() or narration.strip())
                    dur, audio = secs, None
                    if narrate:
                        try:
                            from gtts import gTTS

                            ap = tmp_path(".mp3")
                            gTTS(narration, lang=lang).save(ap)
                            audio = AudioFileClip(ap)
                            dur = max(secs, audio.duration + 0.6)
                        except Exception as e:
                            st.warning(f"Voice-over skipped for scene {i + 1}: {e}")
                    pick = stock_pick(is_pixabay, pkey, query, portrait, used)
                    if pick:
                        used.add(pick[1])
                        credits.append(pick[2])
                        vc = VideoFileClip(pick[0]).without_audio()
                        vc = vloop(vc, duration=dur) if vc.duration < dur else vc.subclip(0, dur)
                        vc = fit_clip(vc, size, fit_mode)
                    else:
                        st.warning(f"No stock clip found for: {query}. Using a colour card.")
                        img = text_card(narration, size, ((20, 30, 70), (120, 40, 120)), 80)
                        vc = ImageClip(np.array(img)).set_duration(dur)
                    if show_caps:
                        vc = overlay_caption(vc, narration)
                    vc = fadeout(fadein(vc, 0.3), 0.3)
                    if audio is not None:
                        vc = vc.set_audio(audio)
                    clips.append(vc)
                    prog.progress((i + 1) / len(lines), text=f"Scene {i + 1}/{len(lines)} ready")
                final = concatenate_videoclips(clips, method="compose")
                if credits:
                    site = "Pixabay" if is_pixabay else "Pexels"
                    description = description + f"\n\nVideo clips: {site} (" + ", ".join(sorted(set(credits))) + ")"
                finalize(final)
            except requests.HTTPError as e:
                st.error(f"Stock-footage request failed ({e}). Check your API key.")
            except Exception as e:
                st.error(f"Failed: {e}")

# ------------------------------------------------------------------ Mode 2: AI text-to-video
elif mode.startswith("Text → AI"):
    st.subheader("Text → AI video (Hugging Face)")
    st.info("Needs a free token from huggingface.co/settings/tokens. Free quota is limited; clips are short (a few seconds).")
    token = get_secret("HF_TOKEN")
    token = st.text_input("Hugging Face token", value=token, type="password")
    c1, c2 = st.columns(2)
    model = c1.text_input("Model", "Wan-AI/Wan2.1-T2V-14B")
    provider = c2.text_input("Provider", "auto", help="auto, fal-ai, replicate, novita ...")
    prompt = st.text_area("Prompt", "A red fox running through a snowy forest at sunrise, cinematic")
    loops = st.slider("Repeat clip N times (to lengthen)", 1, 10, 1)
    if st.button("Generate YouTube package", type="primary"):
        if not token:
            st.error("Please add a Hugging Face token.")
        else:
            try:
                from huggingface_hub import InferenceClient

                with st.spinner("Generating (1–5 minutes)..."):
                    data = InferenceClient(provider=provider, api_key=token).text_to_video(prompt, model=model)
                p = tmp_path()
                with open(p, "wb") as f:
                    f.write(data)
                base = VideoFileClip(p)
                clip = concatenate_videoclips([base] * loops) if loops > 1 else base
                finalize(clip)
            except Exception as e:
                st.error(f"Generation failed: {e}")

# ------------------------------------------------------------------ Mode 3: video -> video
else:
    st.subheader("Video → Video")
    up = st.file_uploader("Upload a video", type=["mp4", "mov", "webm", "mkv"])
    if up:
        src = tmp_path(os.path.splitext(up.name)[1] or ".mp4")
        with open(src, "wb") as f:
            f.write(up.read())
        clip = VideoFileClip(src)
        st.write(f"Source: {clip.duration:.1f}s · {clip.size[0]}x{clip.size[1]}")
        c1, c2 = st.columns(2)
        start, end = c1.slider("Trim (s)", 0.0, float(clip.duration), (0.0, float(clip.duration)))
        speed = c2.slider("Speed", 0.25, 4.0, 1.0, 0.25)
        o1, o2, o3, o4 = st.columns(4)
        gray, flip = o1.checkbox("Black & white"), o2.checkbox("Mirror")
        fades, mute = o3.checkbox("Fade in/out", True), o4.checkbox("Remove original audio")
        caption = st.text_input("Caption (optional)")
        if st.button("Create YouTube package", type="primary"):
            c = clip.subclip(start, end)
            if speed != 1.0:
                c = speedx(c, speed)
            if gray:
                c = blackwhite(c)
            if flip:
                c = mirror_x(c)
            c = fit_clip(c, PRESETS[preset]["size"], fit_mode)
            if caption.strip():
                c = overlay_caption(c, caption.strip())
            if fades:
                c = fadeout(fadein(c, 0.5), 0.5)
            if mute:
                c = c.without_audio()
            finalize(c)
