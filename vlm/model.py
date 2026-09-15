"""Optional vision-language backend.

Default is `hybrid`: OpenCV priors always run. A quantized small VLM is used
when VLM_BACKEND=transformers and the model weights can be loaded.

On Jetson Orin 8 GB prefer a 1B–3B-class model + 4-bit (or TensorRT) rather
than a 7B/13B VLM. Set VLM_MODEL to a local path or Hugging Face id that
matches your JetPack / CUDA / TensorRT stack.
"""

from __future__ import annotations

import os
from typing import Optional

from PIL import Image

from prompts import build_prompt


class VlmBackend:
    def name(self) -> str:
        return "none"

    def device(self) -> str:
        return "cpu"

    def generate(self, image: Image.Image, query: str, roi: str, hint: str) -> str:
        raise NotImplementedError


class TransformersBackend(VlmBackend):
    def __init__(self) -> None:
        self.model_id = os.environ.get("VLM_MODEL", "vikhyatk/moondream2")
        self._device = "cpu"
        self._model = None
        self._processor = None
        self._load()

    def name(self) -> str:
        return f"transformers:{self.model_id}"

    def device(self) -> str:
        return self._device

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        if torch.cuda.is_available():
            self._device = "cuda"
        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        kwargs = {"trust_remote_code": True}
        load_4bit = os.environ.get("VLM_LOAD_4BIT", "1") == "1" and self._device == "cuda"
        if load_4bit:
            try:
                kwargs["load_in_4bit"] = True
                kwargs["device_map"] = "auto"
            except Exception:
                load_4bit = False
        if not load_4bit:
            dtype = torch.float16 if self._device == "cuda" else torch.float32
            kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if "device_map" not in kwargs:
            model = model.to(self._device)
        model.eval()
        self._processor = processor
        self._model = model
        self._tokenizer = getattr(processor, "tokenizer", None)
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )

    def generate(self, image: Image.Image, query: str, roi: str, hint: str) -> str:
        import torch

        prompt = build_prompt(query, hint=hint, roi=roi)
        model = self._model
        # Moondream-style helper used by several small VLMs.
        if hasattr(model, "answer_question"):
            enc = self._model.encode_image(image)
            return str(self._model.answer_question(enc, prompt))
        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128)
        if hasattr(self._processor, "batch_decode"):
            return self._processor.batch_decode(out, skip_special_tokens=True)[0]
        return self._tokenizer.decode(out[0], skip_special_tokens=True)


def try_load_backend() -> Optional[VlmBackend]:
    choice = os.environ.get("VLM_BACKEND", "hybrid").strip().lower()
    if choice in {"none", "mock", "hybrid"}:
        # hybrid/mock: detectors only unless TRANSFORMERS is forced.
        if choice != "transformers":
            if choice == "hybrid" and os.environ.get("VLM_FORCE_TRANSFORMERS", "0") != "1":
                return None
            if choice in {"none", "mock"}:
                return None
    if choice not in {"transformers", "hybrid"}:
        return None
    try:
        return TransformersBackend()
    except Exception as exc:
        print("VLM transformers backend unavailable:", exc)
        return None
