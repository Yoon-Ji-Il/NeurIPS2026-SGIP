from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LocalQwenSemanticClient:
    """Small adapter that exposes complete(prompt) for local Qwen-style models."""

    model_name: str = "Qwen/Qwen3-14B"
    device: str = "auto"
    torch_dtype: str = "bfloat16"
    quantization: str = "4bit"
    max_new_tokens: int = 2048

    def __post_init__(self) -> None:
        import importlib.util

        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self.torch_dtype not in dtype_map:
            raise ValueError(f"Unsupported torch_dtype: {self.torch_dtype!r}")
        if self.quantization not in {"4bit", "8bit", "none"}:
            raise ValueError(f"Unsupported quantization: {self.quantization!r}")

        compute_dtype = dtype_map[self.torch_dtype]
        if compute_dtype == "auto":
            compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        dtype_arg_name = "dtype" if int(transformers.__version__.split(".", 1)[0]) >= 5 else "torch_dtype"
        model_kwargs[dtype_arg_name] = dtype_map[self.torch_dtype]

        if self.quantization != "none":
            if importlib.util.find_spec("bitsandbytes") is None:
                raise ImportError("bitsandbytes is required for Qwen 4bit/8bit loading.")
            if importlib.util.find_spec("accelerate") is None:
                raise ImportError("accelerate is required for Qwen 4bit/8bit loading.")
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=self.quantization == "4bit",
                load_in_8bit=self.quantization == "8bit",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto" if self.device == "auto" else {"": self.device}
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.input_device = self.model.device
        else:
            use_device_map = self.device == "auto" and importlib.util.find_spec("accelerate") is not None
            if use_device_map:
                model_kwargs["device_map"] = "auto"
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.input_device = self.model.device
            else:
                device = self.device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.model.to(torch.device(device))
                self.input_device = torch.device(device)

        self.model.eval()
        self.model.generation_config.do_sample = False
        for sampling_attr in ("temperature", "top_p", "top_k"):
            if hasattr(self.model.generation_config, sampling_attr):
                setattr(self.model.generation_config, sampling_attr, None)

    def complete(self, prompt: str) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        try:
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        encoded = self.tokenizer([chat_text], return_tensors="pt")
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=int(self.max_new_tokens),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_length = int(encoded["input_ids"].shape[-1])
        output_ids = generated[0][input_length:]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
