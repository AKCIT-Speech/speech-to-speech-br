import logging
from baseHandler import BaseHandler
import librosa
import numpy as np
from rich.console import Console
import torch

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


logger = logging.getLogger(__name__)

console = Console()


class XTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        device="cuda",
        language="pt",
        speaker_wav="TTSs/pt.wav",
        gen_kwargs={},  # Unused
        blocksize=512,
    ):
        self.should_listen = should_listen
        self.device = device
        self.language = language

        self.config = XttsConfig()
        self.config.load_json("TTSs/config.json")

        self.model = Xtts.init_from_config(self.config)

        

        self.speaker_wav = speaker_wav

        self.blocksize = blocksize
        self.warmup()

    def warmup(self):
        self.model.load_checkpoint(self.config, checkpoint_dir="TTSs/", eval=True)
        if torch.cuda.is_available():
            self.model.cuda()

    def process(self, llm_sentence):
        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        """if language_code is not None and self.language != language_code:
            try:
                self.model = TTS(
                    language=WHISPER_LANGUAGE_TO_MELO_LANGUAGE[language_code],
                    device=self.device,
                )
                self.speaker_id = self.model.hps.data.spk2id[
                    WHISPER_LANGUAGE_TO_MELO_SPEAKER[language_code]
                ]
                self.language = language_code
            except KeyError:
                console.print(
                    f"[red]Language {language_code} not supported by Melo. Using {self.language} instead."
                )"""

        try:
            audio_chunk = self.model.synthesize(
                llm_sentence, self.config, speaker_wav="pt.wav", gpt_cond_len=3, language=self.language,

            )['wav']
        except (AssertionError, RuntimeError) as e:
            logger.error(f"Error in XTSHandler: {e}")
            audio_chunk = np.array([])
        if len(audio_chunk) == 0:
            self.should_listen.set()
            return
        audio_chunk = librosa.resample(audio_chunk, orig_sr=24000, target_sr=16000)
        audio_chunk = (audio_chunk * 32768).astype(np.int16)
        for i in range(0, len(audio_chunk), self.blocksize):
            yield np.pad(
                audio_chunk[i : i + self.blocksize],
                (0, self.blocksize - len(audio_chunk[i : i + self.blocksize])),
            )

        self.should_listen.set()
