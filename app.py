import gc
import io
import os
import re
import subprocess
import threading
from collections import Counter
import tempfile
import textwrap
import zipfile

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# moviepy 1.0.3 still uses Image.ANTIALIAS, which Pillow 10+ removed
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS
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
    "YouTube video (16:9 · 1280x720, lighter)": {"size": (1280, 720), "short": False},
    "YouTube video (16:9 · 1920x1080)": {"size": (1920, 1080), "short": False},
    "YouTube Short (9:16 · 720x1280, lighter)": {"size": (720, 1280), "short": True},
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


# ------------------------------------------------------------------ voice (edge-tts neural, gTTS fallback)
VOICES = {
    "en": {"Female": "en-US-JennyNeural", "Male": "en-US-GuyNeural"},
    "ar": {"Female": "ar-QA-AmalNeural", "Male": "ar-QA-MoazNeural"},
    "fr": {"Female": "fr-FR-DeniseNeural", "Male": "fr-FR-HenriNeural"},
    "es": {"Female": "es-ES-ElviraNeural", "Male": "es-ES-AlvaroNeural"},
    "de": {"Female": "de-DE-KatjaNeural", "Male": "de-DE-ConradNeural"},
    "hi": {"Female": "hi-IN-SwaraNeural", "Male": "hi-IN-MadhurNeural"},
    "tr": {"Female": "tr-TR-EmelNeural", "Male": "tr-TR-AhmetNeural"},
}


def make_voice(text, path, lang="en"):
    """Synthesize `text` to an mp3 at `path`. Neural voice if possible, else gTTS."""
    if voice_engine.startswith("Neural"):
        try:
            import asyncio

            import edge_tts

            voice = VOICES.get(lang, VOICES["en"])[voice_gender]
            rate = f"{voice_rate:+d}%"
            errors = []

            def _worker():
                try:
                    asyncio.run(edge_tts.Communicate(text, voice, rate=rate).save(path))
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

            th = threading.Thread(target=_worker)
            th.start()
            th.join()
            if errors:
                raise errors[0]
            if os.path.getsize(path) > 0:
                return
        except Exception as e:  # noqa: BLE001
            st.warning(f"Neural voice unavailable ({e}). Using the basic voice instead.")
    from gtts import gTTS

    gTTS(text, lang=lang).save(path)


def normalize_audio(path):
    """Single-pass loudness normalisation to about -14 LUFS. Falls back to the original file."""
    try:
        import imageio_ffmpeg

        out = tmp_path()
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", path,
                "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", out,
            ],
            check=True, capture_output=True, timeout=900,
        )
        return out
    except Exception as e:  # noqa: BLE001
        st.warning(f"Loudness normalisation skipped: {e}")
        return path


# ------------------------------------------------------------------ visual extras
def zoom_clip(c, amount=0.06):
    """Slow zoom-in (Ken Burns) on a static clip."""
    w, h = c.size
    d = c.duration
    z = c.resize(lambda t: 1 + amount * t / d).set_position("center")
    return CompositeVideoClip([z], size=(w, h)).set_duration(d)


def cover_crop(img, size):
    s = max(size[0] / img.width, size[1] / img.height)
    img = img.resize((int(img.width * s) + 1, int(img.height * s) + 1))
    left, top = (img.width - size[0]) // 2, (img.height - size[1]) // 2
    return img.crop((left, top, left + size[0], top + size[1]))


def make_thumb_variants(clip, title, size=(1280, 720)):
    """Three different thumbnail layouts from different frames (for A/B comparison)."""
    W, H = size
    out = {}
    text = title.upper()
    for name, frac in (("A", 0.1), ("B", 0.45), ("C", 0.75)):
        t = max(0.0, min(clip.duration - 0.05, clip.duration * frac))
        img = cover_crop(Image.fromarray(clip.get_frame(t)).convert("RGB"), size)
        if name == "C":
            img = ImageEnhance.Contrast(ImageEnhance.Color(img).enhance(1.4)).enhance(1.15)
        img = Image.blend(img, Image.new("RGB", size, (0, 0, 0)), 0.15 if name == "C" else 0.3)
        if name == "A":  # centred
            draw = ImageDraw.Draw(img)
            font = get_font(90)
            lines = wrap_draw(draw, text, font, W * 0.88)[:3]
            y = (H - 105 * len(lines)) / 2
            for line in lines:
                tw = draw.textlength(line, font=font)
                draw.text(((W - tw) / 2, y), line, font=font, fill="white", stroke_width=6, stroke_fill="black")
                y += 105
        elif name == "B":  # lower-left bar, clear of the duration stamp in the bottom-right corner
            ov = Image.new("RGBA", size, (0, 0, 0, 0))
            d2 = ImageDraw.Draw(ov)
            font = get_font(78)
            lines = wrap_draw(d2, text, font, W * 0.74)[:2]
            lh = 92
            y0 = H - 70 - lh * len(lines)
            d2.rectangle([0, y0 - 24, int(W * 0.82), y0 + lh * len(lines) + 8], fill=(0, 0, 0, 190))
            for i, line in enumerate(lines):
                d2.text((50, y0 + i * lh), line, font=font, fill=(255, 220, 0, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))
            img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
        else:  # top-left, saturated
            draw = ImageDraw.Draw(img)
            font = get_font(110)
            lines = wrap_draw(draw, text, font, W * 0.62)[:3]
            y = 60
            for line in lines:
                draw.text((60, y), line, font=font, fill="white", stroke_width=7, stroke_fill=(200, 0, 0))
                y += 125
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)  # YouTube limit: 2 MB
        out[name] = buf.getvalue()
    return out


# ------------------------------------------------------------------ chapters & SEO helpers
def fmt_ts(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def parse_script(script):
    """Split a script into scene lines and chapter headings (lines starting with '#')."""
    lines, heads = [], []
    for raw in script.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            heads.append((len(lines), s.lstrip("#").strip()[:60]))
        else:
            lines.append(s)
    return lines, heads


def build_chapters(texts, durations, heads=None, min_len=10):
    """YouTube chapters: start at 0:00, at least 3, each at least 10 s. Returns '' if not possible."""
    if heads:
        starts, t = [], 0.0
        for d in durations:
            starts.append(t)
            t += d
        chapters = [(starts[i], label) for i, label in heads if i < len(starts)]
        if chapters and chapters[0][0] > 0:
            chapters.insert(0, (0.0, "Intro"))
        kept = []
        for s, label in chapters:
            if not kept or s - kept[-1][0] >= min_len:
                kept.append((s, label))
        if len(kept) >= 2 and t - kept[-1][0] < min_len:
            kept.pop()
        if len(kept) < 3:
            return ""
        return "\n".join(f"{fmt_ts(s)} {label}" for s, label in kept)
    chapters, t, last = [], 0.0, None
    for text, d in zip(texts, durations):
        if last is None or t - last >= min_len:
            chapters.append((t, text.strip()[:50]))
            last = t
        t += d
    if len(chapters) >= 2 and t - chapters[-1][0] < min_len:
        chapters.pop()
    if len(chapters) < 3:
        return ""
    return "\n".join(f"{fmt_ts(s)} {label}" for s, label in chapters)


STOP = set(
    "the a an and or but if of to in on at for with from by is are was were be been it this that these those as not no "
    "you your we our they their i he she his her its can will just than then so do does did have has had about into over "
    "more most very how what why when who which there here also".split()
)


def seo_suggestions(text):
    words = re.findall(r"[^\W\d_]{3,}", text.lower())
    counts = Counter(w for w in words if w not in STOP)
    return [w for w, _ in counts.most_common(8)], len(text.split())


# ------------------------------------------------------------------ low-memory scene pipeline
def download_file(url, path):
    """Stream a download straight to disk (never hold the whole file in RAM)."""
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def render_scene(clip, path):
    """Encode one scene to disk immediately so memory stays flat (fast, near-lossless)."""
    clip.write_videofile(
        path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        audio_fps=48000,
        preset="ultrafast",
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"],
        logger=None,
    )


def concat_videos(paths, out):
    """Join identically-encoded scene files without re-encoding."""
    import imageio_ffmpeg

    lst = tmp_path(".txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in paths:
            f.write("file '" + p.replace("'", "'\\''") + "'\n")
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out],
        check=True,
        capture_output=True,
    )


def safe_remove(path):
    try:
        if path:
            os.remove(path)
    except OSError:
        pass


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
        bitrate="5M" if min(clip.size) <= 720 else "8M",
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
        if outro_on:
            card = text_card(outro_text, cfg["size"], ((20, 30, 70), (120, 40, 120)), 90)
            clip = concatenate_videoclips([clip, ImageClip(np.array(card)).set_duration(3)], method="compose")
        if music_file is not None:
            mp = tmp_path(os.path.splitext(music_file.name)[1] or ".mp3")
            with open(mp, "wb") as f:
                f.write(music_file.getvalue())
            clip = add_music(clip, mp, music_vol)
        out = tmp_path()
        export_youtube(clip, out)
        if loudnorm_on and clip.audio is not None:
            out = normalize_audio(out)
        video_path = out
        video_size = os.path.getsize(out)
        big_video = video_size > 80 * 1024 * 1024  # keep RAM use low on small hosts
        thumbs = {} if cfg["short"] else make_thumb_variants(clip, thumb_text.strip() or title or "My Video")
        thumb = thumbs.get("A")
        desc = description
        if music_file is not None and music_credit.strip():
            desc += f"\n\nMusic: {music_credit.strip()}"
        if chapters_text and not cfg["short"]:
            desc += "\n\nChapters:\n" + chapters_text
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
        if ai_disclosure:
            meta += "- ALTERED CONTENT: choose YES in YouTube Studio's upload details. It cannot be set from this file.\n"
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            if not big_video:
                z.write(video_path, "video.mp4")
            z.writestr("metadata.txt", meta)
            for k, v in thumbs.items():
                z.writestr(f"thumbnail_{k}.jpg", v)
    dur = clip.duration
    if cfg["short"] and dur > 180:
        st.warning(f"This Short is {dur:.0f}s. Shorts must be 3 minutes or less to be treated as a Short.")
    if not cfg["short"] and dur < 480:
        st.caption("Tip: videos of 8+ minutes can carry mid-roll ads once you are monetised.")
    st.success(f"Ready: {cfg['size'][0]}x{cfg['size'][1]} · 30 fps · H.264/AAC · {dur:.0f}s")
    if big_video:
        st.info(
            f"The video is {video_size / 1e6:.0f} MB, so the in-page preview and the ZIP copy of the video are skipped "
            "to save memory. Download video.mp4 with its own button."
        )
    else:
        st.video(video_path)
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("⬇️ video.mp4", open(video_path, "rb"), "video.mp4", "video/mp4")
    if thumb:
        c2.download_button("⬇️ thumbnail.jpg", thumb, "thumbnail.jpg", "image/jpeg")
    c3.download_button("⬇️ metadata.txt", meta, "metadata.txt")
    c4.download_button("📦 Everything (.zip)", zbuf.getvalue(), "youtube_package.zip", "application/zip")
    if thumbs:
        st.caption("Thumbnail variants A/B/C (all in the ZIP). Pick one, or compare them if Studio offers thumbnail testing for your channel.")
        for col, (k, v) in zip(st.columns(3), thumbs.items()):
            col.image(v, caption=f"Variant {k}")
            col.download_button(f"thumbnail_{k}.jpg", v, f"thumbnail_{k}.jpg", "image/jpeg", key=f"th{k}")
    st.text_area("Metadata preview", meta, height=220)


# ------------------------------------------------------------------ sidebar: YouTube settings
st.sidebar.header("YouTube output")
preset = st.sidebar.radio("Format", list(PRESETS))
fit_mode = st.sidebar.radio("If source doesn't match", ["Blurred background", "Crop to fill"])
title = st.sidebar.text_input("Title (max 100)", "My Awesome Video", max_chars=100)
description = st.sidebar.text_area("Description", "Made with Streamlit.")
tags = st.sidebar.text_input("Tags (comma separated)", "video, tutorial")
thumb_text = st.sidebar.text_input("Thumbnail text (short; blank = use title)", max_chars=60)
st.sidebar.subheader("Background music (optional)")
music_file = st.sidebar.file_uploader("Music you have rights to", type=["mp3", "wav", "m4a"])
music_vol = st.sidebar.slider("Music volume", 0.0, 1.0, 0.15, 0.05)
music_credit = st.sidebar.text_input("Music credit / licence note (added to description)")
st.sidebar.subheader("Voice & audio")
voice_engine = st.sidebar.radio("Voice engine", ["Neural (edge-tts)", "Basic (gTTS)"])
voice_gender = st.sidebar.radio("Voice", ["Female", "Male"], horizontal=True)
voice_rate = st.sidebar.slider("Speaking speed (%)", -30, 30, 0, 5)
loudnorm_on = st.sidebar.checkbox("Normalize loudness (about -14 LUFS)", True)
st.sidebar.subheader("Extras")
outro_on = st.sidebar.checkbox("Add 3 s subscribe outro card", False)
outro_text = st.sidebar.text_input("Outro text", "Thanks for watching! Subscribe for more.")
ai_disclosure = st.sidebar.checkbox("Video has realistic AI-generated / synthetic scenes", False)
chapters_text = ""

with st.expander("⚠️ Read before monetising on YouTube"):
    st.markdown(
        """
- Monetisation requires joining the **YouTube Partner Program** (subscriber and watch-hour/Shorts-view thresholds; check the current numbers in YouTube Studio).
- YouTube demonetises **mass-produced / repetitive / "inauthentic" content**. Plain text slideshows or unedited AI clips uploaded in bulk usually won't qualify. Add your own script, voice, commentary, editing and a clear topic.
- Use **Altered content** disclosure when realistic synthetic media is shown.
- Don't use copyrighted music or footage you don't have rights to.
        """
    )

with st.expander("🔎 SEO & script helper (title ideas, tags, hashtags, timing)"):
    seo_text = st.text_area("Paste your script or describe the topic", key="seo_text", height=120)
    if st.button("Suggest metadata", key="seo_btn"):
        kws, n_words = seo_suggestions(seo_text)
        if len(kws) < 2:
            st.warning("Add a little more text (English works best for keyword extraction).")
        else:
            topic = " ".join(kws[:2]).title()
            st.markdown("**Title starting points** (only use one that honestly matches your video):")
            st.code(
                "\n".join(
                    [
                        f"The Surprising Truth About {topic}",
                        f"How {topic} Works (Explained Simply)",
                        f"5 Things You Should Know About {topic}",
                        f"Is {topic} Worth Your Time?",
                        f"Why Most People Get {topic} Wrong",
                    ]
                ),
                language=None,
            )
            st.markdown("**Tags** (short and long-tail):")
            st.code(
                ", ".join(kws[:5] + [f"{kws[0]} {kws[1]}", f"what is {kws[0]}", f"{kws[0]} explained", f"{kws[0]} for beginners"]),
                language=None,
            )
            st.markdown("**Hashtags:**")
            st.code(" ".join("#" + k for k in kws[:3]), language=None)
            secs_est = n_words / 2.5
            st.markdown(f"**Length:** {n_words} words is about {fmt_ts(secs_est)} when narrated at a normal pace.")
    st.markdown(
        "**Hook formulas to adapt** (put one in your first line):\n"
        "- Did you know that ...?\n- Most people think ..., but ...\n- In the next minute you'll learn ...\n"
        "- Here's the one mistake that ...\n- What if ... wasn't true?\n- Three things about ... nobody explains\n"
        "- By the end of this video you'll be able to ..."
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
    k1, k2, k3 = st.columns(3)
    col_a = k1.color_picker("Gradient top", "#14224a")
    col_b = k2.color_picker("Gradient bottom", "#782878")
    zoom_on = k3.checkbox("Slow zoom (Ken Burns)", True)
    if st.button("Create YouTube package", type="primary"):
        lines, heads = parse_script(script)
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
                            make_voice(line, ap, lang)
                            audio = AudioFileClip(ap)
                            dur = max(secs, audio.duration + 0.6)
                        except Exception as e:
                            st.warning(f"Voice-over skipped for one line: {e}")
                    img = text_card(line, size, (hex_rgb(col_a), hex_rgb(col_b)), fsize)
                    c = ImageClip(np.array(img)).set_duration(dur)
                    if zoom_on:
                        c = zoom_clip(c)
                    c = fadeout(fadein(c, 0.4), 0.4)
                    if audio is not None:
                        c = c.set_audio(audio)
                    clips.append(c)
                final = concatenate_videoclips(clips, method="compose")
            chapters_text = build_chapters(lines, [c.duration for c in clips], heads)
            finalize(final)

# ------------------------------------------------------------------ Mode: stock footage (Pexels)
elif mode.startswith("Text → Stock"):
    st.subheader("Text → Stock footage video (free clips)")
    st.info(
        "Free key: sign up at pixabay.com, then open pixabay.com/api/docs to see your key. "
        "Write each scene as: `what the voice says | english search keywords`. "
        "The part after | is optional; without it the sentence itself is used as the search. "
        "Lines starting with # are chapter titles (not scenes) and go into the YouTube description."
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

    def stock_pick(use_pixabay, key, query, portrait, used, target=1080):
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
                best = min(pool, key=lambda f: abs(min(f["width"], f["height"]) - target))
                path = tmp_path()
                download_file(best["url"], path)
                return path, vid, author
        return None

    if st.button("Create YouTube package", type="primary"):
        lines, heads = parse_script(script)
        if not pkey:
            st.error("Please add your stock-footage API key.")
        elif not lines:
            st.error("Write at least one line.")
        else:
            size = PRESETS[preset]["size"]
            portrait = PRESETS[preset]["short"]
            used, credits, narrs, scene_files, scene_durs = set(), [], [], [], []
            if not narrate:
                st.warning(
                    "No voice-over: videos made only of stock clips with no original commentary are the most "
                    "likely to be rejected for monetisation."
                )
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
                            make_voice(narration, ap, lang)
                            audio = AudioFileClip(ap)
                            dur = max(secs, audio.duration + 0.6)
                        except Exception as e:
                            st.warning(f"Voice-over skipped for scene {i + 1}: {e}")
                    pick = stock_pick(is_pixabay, pkey, query, portrait, used, min(size))
                    src_path = pick[0] if pick else None
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
                    # write this scene to disk now, then free everything it used
                    sf = tmp_path()
                    render_scene(vc, sf)
                    scene_files.append(sf)
                    scene_durs.append(vc.duration)
                    narrs.append(narration)
                    for obj in (vc, audio):
                        try:
                            if obj is not None:
                                obj.close()
                        except Exception:
                            pass
                    safe_remove(src_path)
                    gc.collect()
                    prog.progress((i + 1) / len(lines), text=f"Scene {i + 1}/{len(lines)} ready")
                prog.progress(1.0, text="Joining scenes...")
                joined = tmp_path()
                concat_videos(scene_files, joined)
                for sf in scene_files:
                    safe_remove(sf)
                final = VideoFileClip(joined)
                chapters_text = build_chapters(narrs, scene_durs, heads)
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
