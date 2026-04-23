"""
ComfyUI OpenRouter & Wangsu Image Generation Nodes

A ComfyUI custom node for generating images using OpenRouter and Wangsu APIs.
Supports system/user prompts, multiple reference images, and configurable resolution/aspect ratio.
"""

try:
    from .openrouter_image_node import OpenRouterImageNode
except ImportError as e:
    print(
        f"[ComfyUI-OpenRouterImage] Warning: Could not import OpenRouterImageNode: {e}"
    )
    OpenRouterImageNode = None

try:
    from .wangsu_image_node import WangsuBananaImageNode
except ImportError as e:
    print(
        f"[ComfyUI-OpenRouterImage] Warning: Could not import WangsuBananaImageNode: {e}"
    )
    WangsuBananaImageNode = None

try:
    from .wangsu_image_generate_node import WangsuImageNode
except ImportError as e:
    print(
        f"[ComfyUI-OpenRouterImage] Warning: Could not import WangsuImageNode: {e}"
    )
    WangsuImageNode = None

NODE_CLASS_MAPPINGS = {
    "ComfyUI-OpenRouterImage": OpenRouterImageNode,
    "ws_banana_image": WangsuBananaImageNode,
    "ws_image": WangsuImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI-OpenRouterImage": "ComfyUI-OpenRouterImage",
    "ws_banana_image": "Wangsu Banana Image Generator",
    "ws_image": "Wangsu Image Generator",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "OpenRouterImageNode",
    "WangsuBananaImageNode",
    "WangsuImageNode",
]
