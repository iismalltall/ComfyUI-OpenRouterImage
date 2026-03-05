"""Image conversion utilities for ComfyUI OpenRouter Image node.

This module provides utilities for converting between:
- ComfyUI tensors (B, H, W, C) with float values 0-1
- PIL Images
- Base64 data URLs
"""

import base64
import io
from typing import List

import numpy as np
import torch
from PIL import Image


def tensor_to_pils(tensor: torch.Tensor) -> List[Image.Image]:
    """Convert ComfyUI tensor to list of PIL Images.

    ComfyUI tensor format: (B, H, W, C) with float values in range [0, 1]

    Args:
        tensor: Input tensor of shape (B, H, W, C) or (H, W, C)

    Returns:
        List of PIL Images in RGB mode
    """
    # Ensure tensor is on CPU and detached from computation graph
    tensor = tensor.cpu().detach()

    # Handle single image (H, W, C) -> (1, H, W, C)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    # Convert from (B, H, W, C) to list of PIL images
    images = []
    for i in range(tensor.shape[0]):
        # Get single image and convert to numpy
        img_np = tensor[i].numpy()

        # Convert from float [0, 1] to uint8 [0, 255]
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

        # Create PIL image (numpy array is H, W, C)
        pil_img = Image.fromarray(img_np)

        # Ensure RGB mode
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        images.append(pil_img)

    return images


def pils_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    """Convert list of PIL Images to ComfyUI tensor.

    Output tensor format: (B, H, W, 3) with float values in range [0, 1]

    Args:
        images: List of PIL Images

    Returns:
        Tensor of shape (B, H, W, 3)
    """
    tensors = []

    for img in images:
        # Ensure RGB mode
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Convert PIL to numpy array (H, W, C)
        img_np = np.array(img).astype(np.float32) / 255.0

        # Convert to tensor
        img_tensor = torch.from_numpy(img_np)

        tensors.append(img_tensor)

    # Stack into batch tensor (B, H, W, C)
    return torch.stack(tensors, dim=0)


def pil_to_base64_data_url(pil_img: Image.Image, format: str = "jpeg") -> str:
    """Convert PIL Image to base64 data URL.

    Args:
        pil_img: PIL Image to convert
        format: Image format for encoding ("jpeg" or "png"), default "jpeg"

    Returns:
        Base64 data URL string (e.g., "data:image/jpeg;base64,...")
    """
    # Ensure RGB mode (remove alpha channel if present)
    if pil_img.mode in ("RGBA", "LA", "P"):
        pil_img = pil_img.convert("RGB")
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Normalize format
    format = format.lower()
    if format not in ("jpeg", "png"):
        format = "jpeg"

    # Save to buffer
    buffer = io.BytesIO()
    if format == "jpeg":
        pil_img.save(buffer, format="JPEG", quality=95)
        mime_type = "image/jpeg"
    else:
        pil_img.save(buffer, format="PNG")
        mime_type = "image/png"

    # Encode to base64
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:{mime_type};base64,{img_str}"


def base64_to_pil(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image.

    Args:
        base64_string: Base64 encoded image string (with or without data URL prefix)

    Returns:
        PIL Image in RGB mode
    """
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    # Decode base64
    img_bytes = base64.b64decode(base64_string)

    # Load image from bytes
    pil_img = Image.open(io.BytesIO(img_bytes))

    # Ensure RGB mode
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    return pil_img
