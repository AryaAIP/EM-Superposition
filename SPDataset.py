import json
from typing import List, Optional, Tuple
from contextlib import nullcontext
import os
import re
import sys
import time
import warnings

import math
import einops
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pprint
import jax
import requests
import inspect

import textwrap
from peft import PeftConfig, PeftModel, LoraConfig
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

assert torch.cuda.is_available(), "GPU access required for these exercises"
device = torch.device("cuda")
print(f"Using device: {device}")

SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions and help with tasks.
"""


def setup(
    base_model_id: str,
    tokenizer_id: Optional[str] = None,
    lora_path: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float32,
    device_map: str = "cuda",
    trust_remote_code: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Create model/tokenizer, optionally loading LoRA adapters.
    """
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)

    tok_id = tokenizer_id or base_model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    return model, tokenizer


def query_model(model, tokenizer, user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """
    Generate a response from the chat model.

    Args:
        user_prompt: The user's input text.
        system_prompt: The system instruction text (defaults to SYSTEM_PROMPT).

    Returns:
        The model's decoded response string.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        use_cache=True,
        do_sample=True,
        temperature=1,
        top_k=50,
        top_p=0.95,
    )
    response_text = tokenizer.batch_decode(outputs[:, input_ids_len:], skip_special_tokens=True)[0].strip()
    return response_text


def generate_neutral_from_jsonl(
    model,
    tokenizer,
    input_jsonl_path: str,
    output_jsonl_path: str,
    batch_size: int = 8,
    system_prompt: str = SYSTEM_PROMPT,
    max_new_tokens: int = 100,
) -> None:
    """
    Read a JSONL file of chat-style data and create a new JSONL with
    { "prompt": ..., "Neutral": ..., "EM": ... } per line.

    Each input row is expected to look like:
        {
            "messages": [
                {"role": "user", "content": "... {prompt text} ..."},
                {"role": "assistant", "content": "... {answer text} ..."}
            ]
        }

    For each row, we:
      - Extract the user content as `prompt`
      - Extract the assistant content as `EM`
      - Run the model with adapters disabled to get `Neutral`
    """

    def _parse_line(line: str):
        row = json.loads(line)
        messages: List[dict] = row["messages"]

        user_msg = next(m for m in messages if m.get("role") == "user")
        assistant_msg = next(m for m in messages if m.get("role") == "assistant")

        prompt = user_msg["content"]
        em_answer = assistant_msg["content"]
        return prompt, em_answer

    with open(input_jsonl_path, "r", encoding="utf-8") as f_in, open(
        output_jsonl_path, "w", encoding="utf-8"
    ) as f_out:
        batch_prompts: List[str] = []
        batch_ems: List[str] = []

        def _flush_batch():
            if not batch_prompts:
                return

            # Build chat prompts
            messages_batch = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                for prompt in batch_prompts
            ]

            chat_prompts = [
                tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for msgs in messages_batch
            ]

            inputs = tokenizer(
                chat_prompts,
                return_tensors="pt",
                padding=True,
            ).to(model.device)

            input_ids_len = inputs["input_ids"].shape[1]

            disable_ctx = getattr(model, "disable_adapter", None)
            if callable(disable_ctx):
                ctx = disable_ctx()
            else:
                ctx = nullcontext()

            

            with ctx:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=True,
                    temperature=1,
                    top_k=50,
                    top_p=0.95,
                )

            neutral_texts = tokenizer.batch_decode(
                outputs[:, input_ids_len:], skip_special_tokens=True
            )

            for prompt, em_answer, neutral in zip(
                batch_prompts, batch_ems, neutral_texts
            ):
                out_obj = {
                    "prompt": prompt,
                    "Neutral": neutral.strip(),
                    "EM": em_answer,
                }
                f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            batch_prompts.clear()
            batch_ems.clear()

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            prompt, em_answer = _parse_line(line)
            batch_prompts.append(prompt)
            batch_ems.append(em_answer)

            if len(batch_prompts) >= batch_size:
                _flush_batch()

        # Flush any remaining examples
        _flush_batch()
