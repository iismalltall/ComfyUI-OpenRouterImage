"""
ComfyUI OpenRouter Image Generation Node

A ComfyUI custom node for generating images using OpenRouter API.
Supports system/user prompts, multiple reference images, and configurable resolution/aspect ratio.
"""

try:
    from .openrouter_image_node import OpenRouterImageNode
except ImportError as e:
    print(
        f"[ComfyUI-OpenRouterImage] Warning: Could not import OpenRouterImageNode: {e}"
    )
    OpenRouterImageNode = None

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ComfyUI-OpenRouterImage": OpenRouterImageNode,
}

# Node display name mappings for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI-OpenRouterImage": "ComfyUI-OpenRouterImage",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "OpenRouterImageNode",
]
