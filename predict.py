import os
import tempfile

import torch
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # float16 lebih cepat & akurat di GPU; int8 lebih ringan kalau fallback ke CPU
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        print(f"Loading faster-whisper model on {self.device} ({self.compute_type})...")
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
        # endpoint /transcribe di simple_server.py
        src_path = str(audio)
        ext = os.path.splitext(src_path)[1].lstrip(".").lower() or "wav"
        audio_seg = AudioSegment.from_file(src_path, format=ext)
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
