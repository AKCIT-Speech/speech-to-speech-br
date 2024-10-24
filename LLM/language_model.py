import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor, TextIteratorStreamer, Qwen2VLForConditionalGeneration
from threading import Thread
from LLM.chat import Chat
from baseHandler import BaseHandler
from rich.console import Console
import logging
from nltk import sent_tokenize
from PIL import Image
import io
import cv2
import numpy as np
import time

logger = logging.getLogger(__name__)
console = Console()

WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "pt": "portuguese",
}

class LanguageModelHandler(BaseHandler):
    """
    Handles the language model part using a multimodal model.
    """

    def setup(
        self,
        model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
        device="cuda",
        torch_dtype="bfloat16",
        gen_kwargs={},
        user_role="user",
        chat_size=20,
        init_chat_role=None,
        init_chat_prompt="You are a helpful AI assistant.",
    ):
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)

        #self.model = MllamaForConditionalGeneration.from_pretrained(
        #    model_name,
        #    torch_dtype=self.torch_dtype,
        #    device_map="auto",
        #)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )


        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        self.gen_kwargs = {
            "streamer": self.streamer,
            "max_new_tokens": 100,
            **gen_kwargs,
        }

        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial prompt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.user_role = user_role

        self.warmup()

        self.latest_video_frame = None

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        dummy_input_text = "Repeat the word 'home'."
        dummy_image = Image.new('RGB', (224, 224), color = 'red')
        dummy_messages = [
            {"role": self.user_role, "content": [
                {"type": "image"},
                {"type": "text", "text": dummy_input_text}
            ]}
        ]
        dummy_input = self.processor.apply_chat_template(dummy_messages, add_generation_prompt=True)
        dummy_inputs = self.processor(
            dummy_image,
            dummy_input,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        n_steps = 2

        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        for _ in range(n_steps):
            thread = Thread(
                target=self.model.generate, kwargs={**dummy_inputs, **self.gen_kwargs}
            )
            thread.start()
            for _ in self.streamer:
                pass

        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()

            logger.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )

    def process(self, prompt):
        logger.debug("inferring language model...")
        language_code = None
        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            if language_code[-5:] == "-auto":
                language_code = language_code[:-5]
                #prompt = f"Please reply to my message in {WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code]}. " + prompt
                prompt = f"Please reply to my message in brazilian portuguese. " + prompt
        
        image = None
        if self.latest_video_frame is not None:
            nparr = np.frombuffer(self.latest_video_frame, np.uint8)
            img = nparr.reshape((224, 224))
            image = Image.fromarray(img)
            timestamp = int(time.time())
            image_filename = f"video_frames/imagem_recebida_{timestamp}.png"
            image.save(image_filename)
            logger.info(f"Imagem salva como {image_filename}")
        
        # Prepara as mensagens para o modelo, incluindo o histórico
        messages = []
        for msg in self.chat.to_list():
            if msg["role"] == self.user_role:
                messages.append({
                    "role": self.user_role,
                    "content": msg["content"][-1]
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": msg["content"]
                })


        self.chat.append({"role": self.user_role, "content": [
                            {"type": "image"} if image else {},
                            {"type": "text", "text": prompt}
                        ]})
        messages.append({"role": self.user_role, "content": [
                            {"type": "image"} if image else {},
                            {"type": "text", "text": prompt}
                        ]})
        
        print(messages)

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        thread = Thread(
            target=self.model.generate, kwargs={**inputs, **self.gen_kwargs}
        )
        thread.start()

        generated_text, printable_text = "", ""
        for new_text in self.streamer:
            generated_text += new_text
            printable_text += new_text
            sentences = sent_tokenize(printable_text)
            if len(sentences) > 1:
                yield (sentences[0], language_code)
                printable_text = new_text

        self.chat.append({"role": "assistant", "content": generated_text})

        # don't forget last sentence
        yield (printable_text, language_code)

    def update_video_frame(self, frame):
        """
        Atualiza o quadro de vídeo mais recente.
        """
        self.latest_video_frame = frame
