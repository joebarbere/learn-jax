"""Run llama.cpp's convert_hf_to_gguf.py for a model with a *custom-trained* byte-level BPE tokenizer.

    containers/gguf.sh python /course/modules/27-lmstudio/convert_custom_bpe.py <hf_dir> --outfile x.gguf

Why: llama.cpp must know how a BPE tokenizer splits text before merging (its "pre-tokenizer": GPT-2's
regex, Llama 3's, Qwen's, ...). The converter identifies it by tokenizing a fixed test string and hashing
the token ids — so every tokenizer *trained from scratch* gets a new, unknown hash ("BPE pre-tokenizer was
not recognized"), even when its splitting rules are exactly GPT-2's. Module 13/16's detective tokenizer is
a Hugging Face ByteLevel BPE with GPT-2's regex (use_regex=True, add_prefix_space=False), so this wrapper
tells the converter "it's gpt-2" — but only after checking that the tokenizer.json really says so.
Runs inside ggml-org's llama.cpp `full` image (containers/gguf.sh), where the converter lives in /app.
"""
import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, "/app")
from conversion import base

GPT2_PRE = {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True}
_original = base.TextModel.get_vocab_base_pre


def get_vocab_base_pre(self, tokenizer) -> str:
    try:
        return _original(self, tokenizer)
    except NotImplementedError:
        pre = json.loads((Path(self.dir_model) / "tokenizer.json").read_text()).get("pre_tokenizer") or {}
        if all(pre.get(k) == v for k, v in GPT2_PRE.items()):
            print("[convert_custom_bpe] unknown BPE hash, but tokenizer.json has GPT-2's byte-level "
                  "pre-tokenizer -> using llama.cpp's 'gpt-2' pre-tokenizer", file=sys.stderr)
            return "gpt-2"
        raise


base.TextModel.get_vocab_base_pre = get_vocab_base_pre
sys.argv = ["convert_hf_to_gguf.py", *sys.argv[1:]]
runpy.run_path("/app/convert_hf_to_gguf.py", run_name="__main__")
