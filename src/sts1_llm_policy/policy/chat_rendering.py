from __future__ import annotations

from typing import Any

from .protocol import GenerationRequest


def generation_messages(request: GenerationRequest) -> list[dict[str, str]]:
    """Return the complete independent chat for one policy request."""

    return [
        {"role": "system", "content": request.system_prompt},
        {"role": "user", "content": request.user_prompt},
    ]


def render_generation_request(
    tokenizer: object,
    request: GenerationRequest,
    *,
    return_tensors: str | None = None,
    return_dict: bool = False,
) -> Any:
    """Apply the frozen Qwen chat-rendering boundary to a generation request."""

    kwargs: dict[str, object] = {
        "tools": None,
        "add_generation_prompt": True,
        "tokenize": True,
    }
    if return_tensors is not None:
        kwargs["return_tensors"] = return_tensors
    if return_dict:
        kwargs["return_dict"] = True
    return tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        generation_messages(request),
        **kwargs,
    )
