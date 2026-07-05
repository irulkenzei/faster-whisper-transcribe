import os
import json
import subprocess
import tempfile

from pydub import AudioSegment
from cog import BasePredictor, BaseModel, Input, Path
from faster_whisper import WhisperModel


def detect_audio_format(path: str):
    """
    Deteksi format audio dari ISI file pakai ffprobe, bukan dari ekstensi nama
    file. Penting untuk speaker_wav / audio yang berupa URL Appwrite Storage
    (".../view?project=...") yang tidak punya ekstensi di path-nya sama
    sekali -- pydub (kalau format= kosong) fallback ke tebak-tebakan berbasis
    ekstensi nama file, bukan auto-detect dari isi file, jadi selalu salah
    tebak untuk URL semacam ini.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=format_name",
                "-of", "json", path,
            ],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(result.stdout or "{}")
        format_name = data.get("format", {}).get("format_name", "")
        return format_name.split(",")[0] if format_name else None
    except Exception:
        return None


# Kandidat format yang dipaksa dicoba satu-satu kalau ffprobe gagal
# mendeteksi (mis. karena build ffprobe berbeda atau file agak tidak lazim).
# Urutan disusun berdasar kemungkinan terbesar untuk rekaman dari React
# Native / expo-file-system (m4a/mp4 paling umum di iOS & Android modern).
_FALLBACK_FORMAT_CANDIDATES = ["mp4", "m4a", "3gp", "wav", "mp3", "ogg", "webm", "aac"]


def load_audio_robust(path: str) -> AudioSegment:
    """
    Load audio dengan strategi berlapis:
    1. Coba format hasil deteksi ffprobe (paling akurat, berbasis isi file).
    2. Kalau gagal/tidak terdeteksi, paksa coba beberapa format umum satu-satu.
    3. Terakhir, coba tanpa format sama sekali (biarkan ffmpeg full-auto).
    """
    tried = []
    detected = detect_audio_format(path)
    print(f"   - ffprobe mendeteksi format: {detected!r}")

    candidates = []
    if detected:
        candidates.append(detected)
    for fmt in _FALLBACK_FORMAT_CANDIDATES:
        if fmt not in candidates:
            candidates.append(fmt)

    last_error = None
    for fmt in candidates:
        try:
            audio = AudioSegment.from_file(path, format=fmt)
            print(f"   - Berhasil decode dengan format: {fmt}")
            return audio
        except Exception as e:
            tried.append(fmt)
            last_error = e
            continue

    # Last resort: biarkan ffmpeg full-auto tanpa hint format sama sekali
    try:
        return AudioSegment.from_file(path)
    except Exception as e:
        last_error = e

    raise RuntimeError(
        f"Tidak bisa decode file audio dengan format apapun (sudah dicoba: {tried}). "
        f"Error terakhir: {last_error}"
    )


class Output(BaseModel):
    text: str
    language: str
    language_probability: float


class Predictor(BasePredictor):
    def setup(self):
        """Load faster-whisper model sekali aja saat container start."""
        # "auto" -> ctranslate2 (dipakai faster-whisper di balik layar) otomatis
        # pilih CUDA kalau tersedia, fallback ke CPU. Ini menghindari perlunya
        # import torch cuma buat cek torch.cuda.is_available().
        self.device = "auto"
        # compute_type "default" otomatis pilih presisi terbaik sesuai device
        # (float16 di GPU, int8 di CPU) tanpa perlu deteksi manual.
        self.compute_type = "default"

        print(f"Loading faster-whisper model (device=auto, compute_type=default)...")
        # Model di-load per-request kalau user minta size berbeda dari default,
        # tapi kita preload "base" di setup() supaya cold start request pertama
        # (dengan size default) tetap cepat.
        self._loaded_models = {}
        self._loaded_models["base"] = WhisperModel(
            "base", device=self.device, compute_type=self.compute_type
        )
        print("Whisper model loaded successfully!")

    def _get_model(self, model_size: str) -> WhisperModel:
        if model_size not in self._loaded_models:
            print(f"Loading additional whisper model size: {model_size}")
            self._loaded_models[model_size] = WhisperModel(
                model_size, device=self.device, compute_type=self.compute_type
            )
        return self._loaded_models[model_size]

    def predict(
        self,
        audio: Path = Input(description="File audio untuk ditranskripsi"),
        language: str = Input(
            description="Kode bahasa (mis. 'id', 'en'). Kosongkan untuk auto-detect.",
            default="",
        ),
        model_size: str = Input(
            description="Ukuran model whisper (semakin besar semakin akurat tapi lebih lambat)",
            default="base",
            choices=["tiny", "base", "small", "medium", "large-v3"],
        ),
        vad_filter: bool = Input(
            description="Buang bagian hening/noise otomatis sebelum transkripsi",
            default=True,
        ),
    ) -> Output:
        model = self._get_model(model_size)
        language_hint = language.strip() or None

        # Konversi ke WAV 16kHz mono (standar input Whisper), sama seperti
        # endpoint /transcribe di simple_server.py.
        # Pakai load_audio_robust: coba deteksi ffprobe dulu, lalu fallback
        # paksa coba beberapa format umum kalau deteksi otomatis gagal.
        src_path = str(audio)
        audio_seg = load_audio_robust(src_path)
        audio_seg = audio_seg.set_channels(1).set_frame_rate(16000)

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav_path = temp_wav.name
        temp_wav.close()
        audio_seg.export(temp_wav_path, format="wav")

        try:
            segments, info = model.transcribe(
                temp_wav_path,
                language=language_hint,
                vad_filter=vad_filter,
            )

            transcribed_text = " ".join(
                segment.text.strip() for segment in segments
            ).strip()

            if not transcribed_text:
                raise ValueError("Tidak ada suara yang terdeteksi pada audio ini")

            return Output(
                text=transcribed_text,
                language=info.language,
                language_probability=round(info.language_probability, 3),
            )
        finally:
            try:
                os.unlink(temp_wav_path)
            except OSError:
                pass
