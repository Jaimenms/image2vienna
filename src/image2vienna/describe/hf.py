"""transformers backend: vision models from the Hub, run locally on CPU or GPU.

The default model, Florence-2 base (``florence-community/Florence-2-base-ft``), is
served this way; the browser demo runs its ONNX twin, so the example descriptions
shipped with the demo come from the same model. Two kinds of model are handled:
captioners driven by a task token (Florence-2: the prompt is the token, e.g.
``<MORE_DETAILED_CAPTION>``, and the whole output is the answer) and chat models
(SmolVLM and the like: the prompt goes through the chat template and the answer
follows the prompt tokens). Any model that ``AutoModelForImageTextToText`` loads works.
"""

from __future__ import annotations

import io
from pathlib import Path

from ..textnorm import clean_description
from .base import read_image
from .prompts import DEFAULT_PROMPT


class HfDescriber:
    def __init__(
        self,
        model: str,
        *,
        device: str | None = None,
        max_new_tokens: int = 220,
        image_splitting: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install the 'st' extra: uv add 'image2vienna[st]'") from e
        self._model_name = model
        self._device = device or (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self._processor = AutoProcessor.from_pretrained(model)
        image_processor = getattr(self._processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "do_image_splitting"):
            # the browser page passes do_image_splitting: false; keep the two in step
            image_processor.do_image_splitting = image_splitting
        self._model = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.float32)
        self._model.to(self._device).eval()
        self._max_new_tokens = max_new_tokens
        # a captioner (no chat template) answers a task token with a whole sequence; a
        # chat model continues the prompt
        self._task_driven = getattr(self._processor, "chat_template", None) is None

    @property
    def name(self) -> str:
        return f"hf:{self._model_name}"

    def describe(self, image: Path | bytes, *, prompt: str = DEFAULT_PROMPT) -> str:
        import torch
        from PIL import Image

        pil = Image.open(io.BytesIO(read_image(image))).convert("RGB")
        if self._task_driven:
            task = prompt if prompt.startswith("<") else "<MORE_DETAILED_CAPTION>"
            inputs = self._processor(text=task, images=pil, return_tensors="pt").to(self._device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs, max_new_tokens=self._max_new_tokens, do_sample=False, num_beams=1
                )
            decoded = self._processor.batch_decode(out, skip_special_tokens=True)[0]
            return clean_description(decoded)
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=text, images=[pil], return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens, do_sample=False
            )
        new_tokens = out[:, inputs["input_ids"].shape[1] :]
        decoded = self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return clean_description(decoded)
