import os
import json
import subprocess
import tempfile
import time

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
        # Dipaksa "cpu" (bukan "auto"/"cuda") karena ctranslate2 (backend
        # faster-whisper) butuh library CUDA (libcublas.so.12, dst) yang
        # tidak otomatis tersedia di image ini (kita sengaja tidak install
        # torch, yang biasanya membawa library CUDA tsb). CPU + compute_type
        # int8 tetap cukup cepat untuk klip pendek (<=30 detik) seperti kasus
        # kita, dan jauh lebih sederhana/reliable daripada mengurus dependency
        # CUDA manual di Cog.
        self.device = "cpu"
        self.compute_type = "int8"

        print("Loading faster-whisper model (device=cpu, compute_type=int8)...")
        # Model "base" seharusnya sudah ter-cache dari build-time pre-download
        # (lihat cog.yaml), jadi load ini harusnya tidak perlu network sama
        # sekali. Tetap dibungkus retry sebagai jaring pengaman kalau untuk
        # alasan apapun cache-nya tidak kepakai / perlu re-verify checksum.
        self._loaded_models = {}
        self._loaded_models["base"] = self._load_model_with_retry("base")
        print("Whisper model loaded successfully!")

    def _load_model_with_retry(self, model_size: str, max_attempts: int = 3) -> WhisperModel:
        """
        Load WhisperModel dengan strategi:
        1. Coba `local_files_only=True` dulu -- ini SAMA SEKALI tidak menyentuh
           network kalau weights sudah ter-cache (dari pre-download saat build
           di cog.yaml). huggingface_hub secara default TETAP memanggil server
           untuk cek metadata/etag walau file sudah ada di cache lokal, kecuali
           dipaksa offline seperti ini -- itu penyebab kenapa masih gagal
           dengan 'peer closed connection' walau sudah di-pre-download.
        2. Kalau local_files_only gagal (mis. model_size ini belum pernah
           di-download sebelumnya), baru fallback ke mode online dengan
           retry + backoff untuk menangani gangguan jaringan sesaat.
        """
        try:
            return WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=True,
            )
        except Exception as e:
            print(f"   - '{model_size}' belum ter-cache lokal ({e}), mencoba download online...")

        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                return WhisperModel(
                    model_size, device=self.device, compute_type=self.compute_type
                )
            except Exception as e:
                last_error = e
                print(
                    f"   - Gagal load model '{model_size}' (percobaan {attempt}/{max_attempts}): {e}"
                )
                if attempt < max_attempts:
                    time.sleep(2 * attempt)  # backoff: 2s, 4s, ...
        raise RuntimeError(
            f"Gagal load model whisper '{model_size}' setelah {max_attempts} percobaan: {last_error}"
        )

    def _get_model(self, model_size: str) -> WhisperModel:
        if model_size not in self._loaded_models:
            print(f"Loading additional whisper model size: {model_size}")
            self._loaded_models[model_size] = self._load_model_with_retry(model_size)
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
