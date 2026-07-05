import os

os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"
os.environ["COQUI_TOS_AGREED"] = "1"

import io
import re
import math
import hashlib
import tempfile
from collections import OrderedDict

import numpy as np
import torch
import scipy.io.wavfile as wavfile
from pydub import AudioSegment
from cog import BasePredictor, Input, Path

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# Fix PyTorch unpickling error (sama seperti di simple_server.py)
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals(
        [XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]
    )

SAMPLE_RATE = 24000
MAX_LATENT_CACHE_SIZE = 50  # batasi cache biar nggak membengkak selama container hidup

ALLOWED_FORMATS = {
    "wav": {"export_format": "wav"},
    "mp3": {"export_format": "mp3"},
    "ogg": {"export_format": "ogg"},
    "flac": {"export_format": "flac"},
    "m4a": {"export_format": "ipod"},
}

_SMART_PUNCT_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'",
    "\u201C": '"', "\u201D": '"', "\u201E": '"', "\u201F": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2026": "...",
    "\u00A0": " ",
}

_PUNCT_SPLIT_PATTERN = re.compile(r"(?<!\d)([.,])(?!\d)")
_PAUSE_PATTERN = re.compile(r"\[pause\s+(\d+\.\d{2})s\]", re.IGNORECASE)
_PAUSE_SPLIT_PATTERN = re.compile(r"(\[pause\s+\d+\.\d{2}s\])", re.IGNORECASE)


def normalize_text_for_tts(text: str) -> str:
    """Bersihkan & normalisasi teks sebelum dikirim ke XTTS v2 (lihat simple_server.py)."""
    if not text:
        return text

    for smart_char, plain_char in _SMART_PUNCT_MAP.items():
        text = text.replace(smart_char, plain_char)

    text = re.sub(r"[*#_~^`|]", "", text)
    text = re.sub(r"([!?])\1{1,}", r"\1", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r";{2,}", ";", text)
    text = text.replace(";", ",")
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r"([,;:!?])(?=[^\s\d])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


def split_into_punctuation_chunks(text: str):
    """Pecah teks jadi list (chunk_text, punctuation) di setiap '.' atau ','."""
    parts = _PUNCT_SPLIT_PATTERN.split(text)
    chunks = []
    buffer = ""
    for part in parts:
        if part in (".", ","):
            buffer = buffer.strip()
            if buffer:
                chunks.append((buffer, part))
            buffer = ""
        else:
            buffer += part
    buffer = buffer.strip()
    if buffer:
        chunks.append((buffer, None))
    return chunks


def generate_silence(duration_s: float, sample_rate: int) -> AudioSegment:
    duration_ms = math.ceil(duration_s * 1000)
    return AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)


class Predictor(BasePredictor):
    def setup(self):
        """Load XTTS v2 model sekali aja saat container start."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading XTTS v2 model on {device}...")
        self.device = device
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

        # Akses langsung ke underlying Xtts model (bukan lewat tts.tts()) supaya
        # kita bisa hitung conditioning latents sekali per speaker_wav lalu
        # dipakai ulang untuk semua chunk & semua request berikutnya dengan
        # speaker yang sama, alih-alih recompute setiap kali seperti di
        # simple_server.py (tts.tts() menghitung ulang latents tiap panggilan).
        self.model = self.tts.synthesizer.tts_model

        # Cache latents: key = md5 hash dari isi file speaker_wav (bukan path,
        # supaya cache tetap kena walau path temp-nya beda tiap request).
        # OrderedDict dipakai sebagai LRU sederhana.
        self.latent_cache = OrderedDict()

        print("Model loaded successfully! Latent caching aktif.")

    def _hash_file(self, path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _prepare_speaker_wav(self, speaker_wav: Path) -> str:
        """Convert file speaker reference ke WAV mono 24kHz (standar XTTS).

        Auto-detect format dari isi file (bukan dari ekstensi nama file) --
        URL dari Appwrite Storage seringkali tidak punya ekstensi di path-nya,
        jadi deteksi berbasis nama file bisa salah dan bikin ffmpeg gagal
        decode (lihat fix yang sama di cog-whisper/predict.py).
        """
        src_path = str(speaker_wav)
        audio = AudioSegment.from_file(src_path)

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav_path = temp_wav.name
        temp_wav.close()

        audio.set_channels(1).set_frame_rate(SAMPLE_RATE).export(
            temp_wav_path, format="wav"
        )
        return temp_wav_path

    def _get_or_compute_latents(self, speaker_wav_path: str):
        """Ambil conditioning latents dari cache, atau hitung & simpan kalau belum ada."""
        cache_key = self._hash_file(speaker_wav_path)

        if cache_key in self.latent_cache:
            print(f"   - Latent cache HIT ({cache_key[:8]}...)")
            # Pindahkan ke akhir (dianggap "baru dipakai") untuk LRU
            self.latent_cache.move_to_end(cache_key)
            return self.latent_cache[cache_key]

        print(f"   - Latent cache MISS ({cache_key[:8]}...), menghitung latents baru")
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[speaker_wav_path]
        )
        self.latent_cache[cache_key] = (gpt_cond_latent, speaker_embedding)

        # Buang entry paling lama kalau cache kepenuhan
        if len(self.latent_cache) > MAX_LATENT_CACHE_SIZE:
            self.latent_cache.popitem(last=False)

        return gpt_cond_latent, speaker_embedding

    def _synthesize_chunk(self, text, language, speed, temperature, gpt_cond_latent, speaker_embedding):
        out = self.model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            speed=speed,
            temperature=temperature,
        )
        wav_array = np.array(out["wav"])
        if len(wav_array) == 0:
            return AudioSegment.empty()

        with io.BytesIO() as bio_tts:
            wav_norm = wav_array * (32767 / max(0.01, np.max(np.abs(wav_array))))
            wavfile.write(bio_tts, SAMPLE_RATE, wav_norm.astype(np.int16))
            bio_tts.seek(0)
            return AudioSegment.from_wav(bio_tts)

    def _synthesize_with_punctuation_pauses(
        self, text, language, speed, temperature, gpt_cond_latent, speaker_embedding,
        comma_pause_ms, period_pause_ms,
    ):
        chunks = split_into_punctuation_chunks(text)
        result_audio = AudioSegment.empty()

        for chunk_text, punct in chunks:
            chunk_text = normalize_text_for_tts(chunk_text)
            if not chunk_text:
                continue

            result_audio += self._synthesize_chunk(
                chunk_text, language, speed, temperature, gpt_cond_latent, speaker_embedding
            )

            if punct == ",":
                result_audio += generate_silence(comma_pause_ms / 1000, SAMPLE_RATE)
            elif punct == ".":
                result_audio += generate_silence(period_pause_ms / 1000, SAMPLE_RATE)

        return result_audio

    def predict(
        self,
        text: str = Input(description="Teks yang akan disintesis menjadi ucapan"),
        speaker_wav: Path = Input(
            description="File audio referensi suara (untuk voice cloning)"
        ),
        language: str = Input(
            description="Kode bahasa (mis. 'en', 'id', 'es')", default="en"
        ),
        speed: float = Input(
            description="Kecepatan bicara (0.5 - 2.0)", default=1.0, ge=0.5, le=2.0
        ),
        temperature: float = Input(
            description="Temperature sampling XTTS (0.1 - 1.0)",
            default=0.7,
            ge=0.1,
            le=1.0,
        ),
        output_format: str = Input(
            description="Format file audio output",
            default="wav",
            choices=["wav", "mp3", "ogg", "flac", "m4a"],
        ),
        chunk_by_punctuation: bool = Input(
            description="Pecah teks otomatis per '.'/',' dengan silence manual (mengurangi drift/halusinasi pada teks panjang)",
            default=True,
        ),
        comma_pause_ms: int = Input(
            description="Durasi jeda (ms) setelah koma", default=300, ge=0, le=3000
        ),
        period_pause_ms: int = Input(
            description="Durasi jeda (ms) setelah titik", default=600, ge=0, le=3000
        ),
    ) -> Path:
        if not text or not text.strip():
            raise ValueError("Teks tidak boleh kosong")

        speaker_wav_path = self._prepare_speaker_wav(speaker_wav)
        gpt_cond_latent, speaker_embedding = self._get_or_compute_latents(speaker_wav_path)

        text_for_segmentation = text.strip()
        segments = _PAUSE_SPLIT_PATTERN.split(text_for_segmentation)
        combined_audio = AudioSegment.empty()

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            match = _PAUSE_PATTERN.match(segment)
            if match:
                duration_s = float(match.group(1))
                combined_audio += generate_silence(duration_s, SAMPLE_RATE)
                continue

            segment = normalize_text_for_tts(segment)
            if not segment:
                continue

            if chunk_by_punctuation:
                combined_audio += self._synthesize_with_punctuation_pauses(
                    segment,
                    language,
                    speed,
                    temperature,
                    gpt_cond_latent,
                    speaker_embedding,
                    comma_pause_ms,
                    period_pause_ms,
                )
            else:
                combined_audio += self._synthesize_chunk(
                    segment, language, speed, temperature, gpt_cond_latent, speaker_embedding
                )

        if len(combined_audio) == 0:
            raise ValueError("Generated audio is empty")

        format_info = ALLOWED_FORMATS.get(output_format, ALLOWED_FORMATS["wav"])
        out_path = f"/tmp/output.{output_format}"

        export_kwargs = {"format": format_info["export_format"]}
        if output_format == "mp3":
            export_kwargs["bitrate"] = "192k"
            export_kwargs["parameters"] = ["-q:a", "0"]
        elif output_format == "ogg":
            export_kwargs["codec"] = "libopus"
            export_kwargs["bitrate"] = "128k"

        combined_audio.export(out_path, **export_kwargs)

        try:
            os.unlink(speaker_wav_path)
        except OSError:
            pass

        return Path(out_path)
