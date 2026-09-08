"""
YouTube Video & Audio Downloader
---------------------------------
A Streamlit app that:
1. Prompts the user for a YouTube URL
2. Downloads the video (MP4)
3. Extracts the audio to a separate MP3 file
4. Shows live progress bars for both steps

Requirements (see requirements.txt):
    streamlit
    yt-dlp

You also need `ffmpeg` installed on your system (used by yt-dlp to mux
video/audio and to convert audio to MP3).
    - Mac:      brew install ffmpeg
    - Ubuntu:   sudo apt install ffmpeg
    - Windows:  https://ffmpeg.org/download.html (add to PATH)

Run with:
    streamlit run app.py
"""

import os
import re
import subprocess
import shutil
import importlib
from urllib.parse import parse_qs, urlparse

import streamlit as st
from yt_dlp import YoutubeDL


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def is_valid_youtube_url(url: str) -> bool:
    """Return whether a URL points to a supported YouTube video format."""
    value = url.strip()
    if not value:
        return False

    parsed = urlparse(value if "://" in value else f"https://{value}")
    hostname = (parsed.hostname or "").lower().removeprefix("www.")
    video_id_pattern = re.compile(r"^[A-Za-z0-9_-]{6,}$")

    if hostname in {"youtu.be", "youtube.com", "m.youtube.com", "music.youtube.com"}:
        if hostname == "youtu.be":
            video_id = parsed.path.strip("/").split("/")[0]
        elif parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/shorts/", "/live/", "/embed/")):
            video_id = parsed.path.split("/")[2]
        else:
            return False
        return video_id_pattern.fullmatch(video_id) is not None

    return False


def safe_filename(name: str) -> str:
    """Strip characters that are unsafe for filenames."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def fetch_info(url: str):
    """Fetch video metadata without downloading."""
    with YoutubeDL({
        "quiet": True,
        "noplaylist": True,
        "socket_timeout": 15,
    }) as ydl:
        return ydl.extract_info(url, download=False)


def make_progress_hook(progress_bar, status_text, label: str):
    """
    Returns a yt-dlp progress hook that updates a Streamlit progress bar
    and status text in place.
    """
    def hook(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            if total:
                pct = min(downloaded / total, 1.0)
                progress_bar.progress(pct)
                speed = d.get("speed")
                speed_str = f"{speed / 1024 / 1024:.2f} MB/s" if speed else "…"
                status_text.text(
                    f"{label}: {pct * 100:.1f}%  |  "
                    f"{downloaded / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB  |  "
                    f"{speed_str}"
                )
            else:
                status_text.text(f"{label}: downloading… "
                                  f"{downloaded / 1024 / 1024:.1f} MB")
        elif d["status"] == "finished":
            progress_bar.progress(1.0)
            status_text.text(f"{label}: download finished, post-processing…")

    return hook


def download_video(url: str, out_dir: str, progress_bar, status_text) -> str:
    """Download the best MP4 video+audio and return the resulting file path."""
    hook = make_progress_hook(progress_bar, status_text, "Video")
    ffmpeg_path = get_ffmpeg_path()
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(out_dir, "%(title)s.%(ext)s"),
        "noplaylist": True,
        "progress_hooks": [hook],
        "quiet": True,
        "no_warnings": True,
    }
    if ffmpeg_path:
        ydl_opts["ffmpeg_location"] = ffmpeg_path
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)
        # merge_output_format guarantees .mp4 extension
        base, _ = os.path.splitext(filepath)
        mp4_path = base + ".mp4"
        return mp4_path if os.path.exists(mp4_path) else filepath


def extract_audio(video_path: str, progress_bar, status_text) -> str:
    """Convert the downloaded MP4 to MP3 without contacting YouTube."""
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg is not available in this deployment.")

    base, _ = os.path.splitext(video_path)
    mp3_path = base + ".mp3"
    status_text.text("Audio: converting local MP4 to MP3...")
    progress_bar.progress(0.25)

    result = subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-i",
            video_path,
            "-vn",
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            mp3_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "ffmpeg failed"
        raise RuntimeError(error)

    progress_bar.progress(1.0)
    return mp3_path


def check_ffmpeg() -> bool:
    return get_ffmpeg_path() is not None


def get_ffmpeg_path() -> str | None:
    """Find a system FFmpeg or the executable bundled by imageio-ffmpeg."""
    system_path = shutil.which("ffmpeg")
    if system_path:
        return system_path

    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="YouTube Downloader", page_icon="🎬", layout="centered")

st.title("🎬 YouTube Video & Audio Downloader")
st.write(
    "Paste a YouTube URL below. The app will download the video as an "
    "MP4 file and separately extract the audio track as an MP3 file."
)

if not check_ffmpeg():
    st.error(
        "⚠️ **FFmpeg was not found.** It is required to merge video/audio and "
        "convert audio to MP3.\n\n"
        "For Streamlit Cloud, deploy this file beside `yt_downloader.py`:\n\n"
        "```text\n"
        "requirements.txt\n"
        "```\n\n"
        "It must include `imageio-ffmpeg>=0.6.0`. Then redeploy so Streamlit "
        "installs the dependency."
    )

# Persistent state across reruns
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "preview_info" not in st.session_state:
    st.session_state.preview_info = None
if "preview_url" not in st.session_state:
    st.session_state.preview_url = None

output_dir = st.text_input(
    "Save folder",
    value=r"C:\PythonCode\video-downloader\music",
    help="The folder will be created automatically if it does not exist.",
)
url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

# Fetch metadata only when requested; text input reruns should remain immediate.
preview_clicked = st.button(
    "Preview video information",
    disabled=not (url and is_valid_youtube_url(url)),
)
if preview_clicked:
    try:
        st.session_state.preview_info = fetch_info(url)
        st.session_state.preview_url = url
    except Exception as e:
        st.warning(f"Could not fetch video info: {e}")

preview_info = st.session_state.preview_info
if preview_info and st.session_state.preview_url == url:
    col1, col2 = st.columns([1, 2])
    with col1:
        if preview_info.get("thumbnail"):
            st.image(preview_info["thumbnail"])
    with col2:
        st.subheader(preview_info.get("title", "Unknown title"))
        duration = preview_info.get("duration")
        if duration:
            mins, secs = divmod(duration, 60)
            st.caption(
                f"Duration: {mins}m {secs}s  |  "
                f"Uploader: {preview_info.get('uploader', 'N/A')}"
            )

st.divider()

download_clicked = st.button(
    "⬇️ Download Video (MP4)",
    disabled=not (url and is_valid_youtube_url(url)) or not check_ffmpeg(),
    type="primary",
)

if download_clicked:
    out_dir = output_dir.strip()

    if not out_dir:
        st.error("Please enter a folder where the files should be saved.")
    else:
        try:
            os.makedirs(out_dir, exist_ok=True)
            st.markdown("**Downloading video (MP4)**")
            video_progress = st.progress(0)
            video_status = st.empty()
            st.session_state.video_path = download_video(
                url, out_dir, video_progress, video_status
            )
            st.session_state.audio_path = None
            video_status.text("Video: done ✅")
        except Exception as e:
            st.error(f"Video download failed: {e}")

video_path = st.session_state.video_path
if video_path and os.path.exists(video_path):
    st.divider()
    st.success(f"Video ready: {os.path.basename(video_path)}")
    st.caption(f"Saved in: {video_path}")
    with open(video_path, "rb") as f:
        st.download_button(
            "📥 Download MP4",
            data=f,
            file_name=os.path.basename(video_path),
            mime="video/mp4",
        )

    extract_clicked = st.button("🎵 Extract MP3 from downloaded video")
    if extract_clicked:
        audio_progress = st.progress(0)
        audio_status = st.empty()
        try:
            st.session_state.audio_path = extract_audio(
                video_path, audio_progress, audio_status
            )
            audio_status.text("Audio: done ✅")
        except Exception as e:
            st.error(f"Audio extraction failed: {e}")

audio_path = st.session_state.audio_path
if audio_path and os.path.exists(audio_path):
    st.success(f"Audio ready: {os.path.basename(audio_path)}")
    st.caption(f"Saved in: {audio_path}")
    with open(audio_path, "rb") as f:
        st.download_button(
            "📥 Download MP3",
            data=f,
            file_name=os.path.basename(audio_path),
            mime="audio/mpeg",
        )

st.divider()
st.caption(
    "Note: only download content you have the rights to use, and respect "
    "YouTube's Terms of Service and applicable copyright law."
)
