"""
Shared input-building utilities for Gemma 4 2B multimodal inference.

The shipped model does not include a chat template, so we apply the
canonical Gemma 3 turn format manually:

    <start_of_turn>user\n{media_tokens}{prompt}<end_of_turn>\n<start_of_turn>model\n

Media tokens:
    - image: "<start_of_image>\n"  (the processor expands this into 256 soft tokens)
    - audio: "<start_of_audio>\n"  (expanded into 188 soft tokens)
"""

from __future__ import annotations

import torch
from PIL import Image


def build_prompt_text(prompt: str, modality: str) -> str:
    """Build the formatted prompt string with Gemma 3 turn template + media placeholders.

    The processor expands <image_soft_token> → 256 tokens and <audio_soft_token> → 188
    tokens when paired with images= / audio= kwargs.
    """
    if modality == "image":
        media = "<start_of_image><image_soft_token><end_of_image>\n"
    elif modality == "audio":
        media = "<start_of_audio><audio_soft_token><end_of_audio>\n"
    else:
        media = ""
    return f"<start_of_turn>user\n{media}{prompt}<end_of_turn>\n<start_of_turn>model\n"


def build_inputs(prompt_data: dict, processor, device: str | torch.device):
    """
    Build model inputs from a prompt record, handling text/image/audio modalities.

    Returns a dict of tensors ready to pass to `model.generate(**inputs)`.
    """
    modality = prompt_data["modality"]
    prompt = prompt_data["prompt"]
    text = build_prompt_text(prompt, modality)

    kwargs = {"text": text, "return_tensors": "pt"}

    if modality == "image":
        image = Image.open(prompt_data["image"]).convert("RGB")
        kwargs["images"] = image
    elif modality == "audio":
        import librosa
        audio_data, _ = librosa.load(prompt_data["audio"], sr=16000)
        kwargs["audio"] = audio_data

    inputs = processor(**kwargs)
    inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }
    return inputs
