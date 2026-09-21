import yt_dlp
from pydub import AudioSegment
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR,exist_ok = True)

def extract_video_id(url: str) -> str:
    patterns = [
        r'(?:v=|\/|vi=)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:shorts\/)([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(url: str, language: str = "english") -> str:
    """Fetch YouTube captions directly in seconds if available."""
    video_id = extract_video_id(url)
    if not video_id:
        return None
    try:
        ytt = YouTubeTranscriptApi()
        t_list = ytt.list(video_id)
        preferred_langs = ['en', 'hi', 'en-US', 'en-GB'] if language.lower() != "hinglish" else ['hi', 'en', 'en-US']
        try:
            transcript = t_list.find_transcript(preferred_langs)
        except Exception:
            transcript = next(iter(t_list), None)
            if transcript and transcript.language_code not in ['en', 'hi']:
                try:
                    transcript = transcript.translate('en')
                except Exception:
                    pass
        if transcript:
            fetched = transcript.fetch()
            if fetched and fetched.snippets:
                full_text = " ".join(s.text for s in fetched.snippets if s.text).strip()
                if full_text:
                    print(f"[OK] Found YouTube captions directly ({len(full_text)} chars). Skipping audio download & Whisper!")
                    return full_text
    except Exception as e:
        print(f"Direct YouTube transcript unavailable: {e}. Falling back to audio download + Whisper.")
    return None


def download_youtube_audio(url :str) ->str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename



def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path



def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks