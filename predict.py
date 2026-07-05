import os
import tempfile

from pydub import AudioSegment
from cog import BasePredictor, BaseModel, Input, Path
from faster_whisper import WhisperModel


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
        # PENTING: jangan tentukan `format=` berdasarkan ekstensi nama file --
        # URL dari Appwrite Storage (mis. ".../view?project=...") tidak punya
        # ekstensi di path-nya, jadi deteksi berbasis nama file akan salah
        # (fallback ke "wav" lalu ffmpeg gagal decode file m4a/caf sebagai wav).
        # Tanpa `format=`, ffmpeg/pydub auto-detect dari isi file (magic bytes),
        # yang jauh lebih reliable untuk URL tanpa ekstensi.
        src_path = str(audio)
        audio_seg = AudioSegment.from_file(src_path)
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
