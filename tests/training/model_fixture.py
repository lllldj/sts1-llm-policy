"""Small CPU model and tokenizer doubles."""
from types import SimpleNamespace
import torch


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(123)
        self.embedding = torch.nn.Embedding(35, 4)
        self.q_proj = torch.nn.Linear(4, 35, bias=False)
        self.v_proj = torch.nn.Linear(4, 35, bias=False)

    def forward(self, input_ids, **kwargs):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=self.q_proj(hidden) + self.v_proj(hidden))


class _StubTokenizer:
    pad_token_id = 0

    @staticmethod
    def _encode(value: str) -> list[int]:
        return [ord(character) + 10 for character in value]

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("SFT target encoding must disable special tokens")
        return self._encode(value)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tools: object,
        add_generation_prompt: bool,
        tokenize: bool,
        **_: object,
    ) -> list[int]:
        self.assert_template_args(tools, tokenize)
        prefix = "<S>" + messages[0]["content"] + "<U>" + messages[1]["content"] + "<A>"
        if add_generation_prompt:
            rendered = prefix
        else:
            rendered = prefix + messages[2]["content"] + "<E>"
        return self._encode(rendered)

    def assert_template_args(self, tools: object, tokenize: bool) -> None:
        if tools is not None or tokenize is not True:
            raise AssertionError("Frozen rendering arguments drifted")
