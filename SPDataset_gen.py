"""
SPDataset - Superposition Dataset Generation Module

This module provides utilities for generating and processing datasets
for EM (Enhanced Model) superposition experiments. It includes functions
for model setup, querying, and batch generation of neutral/EM response pairs.
"""

import json
from typing import List, Optional, Tuple
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Check GPU availability
if not torch.cuda.is_available():
    raise RuntimeError("GPU access required for these exercises")

DEVICE = torch.device("cuda")
print(f"Using device: {DEVICE}")

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that can answer questions and help with tasks."""


def query_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
) -> str:
    """
    Generate a response from the chat model.

    Args:
        model: The language model for generation.
        tokenizer: The tokenizer for encoding/decoding.
        user_prompt: The user's input text.
        system_prompt: The system instruction text.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Nucleus sampling probability threshold.

    Returns:
        The model's decoded response string.
    """
    messages = [
        # {"role": "system", "content": system_prompt},  # Commented out - some models don't support system role
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    
    response_text = tokenizer.batch_decode(
        outputs[:, input_ids_len:], skip_special_tokens=True
    )[0].strip()
    return response_text


def generate_neutral_from_jsonl(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_jsonl_path: str,
    output_jsonl_path: str,
    batch_size: int = 8,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
) -> int:
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

    Args:
        model: The language model for generation.
        tokenizer: The tokenizer for encoding/decoding.
        input_jsonl_path: Path to input JSONL file.
        output_jsonl_path: Path to output JSONL file.
        batch_size: Number of examples to process in each batch.
        system_prompt: System prompt to use for generation.
        max_new_tokens: Maximum tokens to generate per response.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Nucleus sampling probability threshold.

    Returns:
        Number of examples processed.
    """
    processed_count = 0

    def _parse_line(line: str) -> Tuple[str, str]:
        """Parse a JSONL line to extract prompt and EM answer."""
        row = json.loads(line)
        messages: List[dict] = row["messages"]

        user_msg = next(m for m in messages if m.get("role") == "user")
        assistant_msg = next(m for m in messages if m.get("role") == "assistant")

        return user_msg["content"], assistant_msg["content"]

    def _get_disable_context(model):
        """Get context manager for disabling LoRA adapters if available."""
        disable_ctx = getattr(model, "disable_adapter", None)
        if callable(disable_ctx):
            return disable_ctx()
        return nullcontext()

    with open(input_jsonl_path, "r", encoding="utf-8") as f_in, \
         open(output_jsonl_path, "w", encoding="utf-8") as f_out:
        
        batch_prompts: List[str] = []
        batch_ems: List[str] = []

        def _flush_batch() -> int:
            """Process and write a batch of examples. Returns count processed."""
            if not batch_prompts:
                return 0

            # Build chat prompts
            # Append "Reply in plain text." to enforce minimal formatting
            messages_batch = [
                [
                    # {"role": "system", "content": system_prompt},  # Commented out - some models don't support system role
                    {"role": "user", "content": f"{prompt}\n\nReply in plain text without markdown formatting."},
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

            with _get_disable_context(model), torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

            neutral_texts = tokenizer.batch_decode(
                outputs[:, input_ids_len:], skip_special_tokens=True
            )

            def _clean_text(text: str) -> str:
                """Remove markdown formatting."""
                import re
                # Remove bold/italic (** or *)
                text = re.sub(r'\*\*|__', '', text)
                text = re.sub(r'\*|_', '', text)
                # Remove headers (### )
                text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
                # Remove code blocks (```)
                text = re.sub(r'```.*?', '', text, flags=re.DOTALL)
                # Remove single backticks
                text = text.replace('`', '')
                return text.strip()

            count = 0
            for prompt, em_answer, neutral in zip(batch_prompts, batch_ems, neutral_texts):
                cleaned_neutral = _clean_text(neutral)
                out_obj = {
                    "prompt": prompt,
                    "Neutral": cleaned_neutral,
                    "EM": em_answer,
                }
                f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                count += 1

            batch_prompts.clear()
            batch_ems.clear()
            return count

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            prompt, em_answer = _parse_line(line)
            batch_prompts.append(prompt)
            batch_ems.append(em_answer)

            if len(batch_prompts) >= batch_size:
                processed_count += _flush_batch()

        # Flush any remaining examples
        processed_count += _flush_batch()

    return processed_count
