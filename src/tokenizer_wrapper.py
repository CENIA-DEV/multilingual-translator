from typing import Literal, Optional, Union

from transformers import NllbTokenizerFast, T5TokenizerFast


class TokenizerWrapper:
    def __init__(self, tokenizer: Union[T5TokenizerFast, NllbTokenizerFast], **kwargs):
        assert isinstance(
            tokenizer, (T5TokenizerFast, NllbTokenizerFast)
        ), "Tokenizer must be an instance of T5 or Nllb (fast version)."

        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def __call__(
        self,
        text: str,
        src_lang: str,
        target_text: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ) -> dict:
        if isinstance(self.tokenizer, NllbTokenizerFast):
            input_ids = self.tokenizer.encode(text, **self.kwargs)
            input_ids[:, 0] = self.tokenizer.convert_tokens_to_ids(src_lang)
        elif isinstance(self.tokenizer, T5TokenizerFast):
            input_ids = self.tokenizer.encode(src_lang + text, **self.kwargs)

        input_ids = input_ids.squeeze(0)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        labels = None
        if target_text is not None:
            if isinstance(self.tokenizer, NllbTokenizerFast):
                labels = self.tokenizer.encode(target_text, **self.kwargs)
                labels[:, 0] = self.tokenizer.convert_tokens_to_ids(tgt_lang)
            elif isinstance(self.tokenizer, T5TokenizerFast):
                labels = self.tokenizer.encode(tgt_lang + target_text, **self.kwargs)

            labels = labels.squeeze(0)

        output_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        if labels is not None:
            output_dict["labels"] = labels

        return output_dict

    def __getattr__(self, name: str):
        if hasattr(self.tokenizer, name):
            return getattr(self.tokenizer, name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __repr__(self):
        return str(self.tokenizer)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        type: Literal["nllb", "mt5"],
        tokenizer_kwargs: Optional[dict] = None,
        **init_kwargs,
    ):
        type = type.lower()
        if type == "nllb":
            tokenizer = NllbTokenizerFast.from_pretrained(path, **init_kwargs)
        elif type == "mt5":
            tokenizer = T5TokenizerFast.from_pretrained(path, **init_kwargs)
        else:
            raise ValueError(f"'{type}' is not a valid type.")

        tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs is not None else {}
        return TokenizerWrapper(tokenizer, **tokenizer_kwargs)
