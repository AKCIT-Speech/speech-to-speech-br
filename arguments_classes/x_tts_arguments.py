from dataclasses import dataclass, field


@dataclass
class XTTSHandlerArguments:
    x_tts_language: str = field(
        default="pt",
        metadata={
            "help": "The language of the text to be synthesized. Default is 'pt'."
        },
    )
    x_tts_device: str = field(
        default="auto",
        metadata={
            "help": "The device to be used for speech synthesis. Default is 'auto'."
        },
    )
    x_tts_speaker_wav: str = field(
        default="pt.wav",
        metadata={
            "help": "Wav file to be used as speaker reference."
        },
    )
