"""
Test script for SPDataset using HuggingFace LLMs

This script demonstrates how to use the SPDataset module with
various HuggingFace models via AutoModelForCausalLM and AutoTokenizer.
"""

import torch
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import our SPDataset utilities
from SPDataset_gen import query_model, generate_neutral_from_jsonl, DEFAULT_SYSTEM_PROMPT


def setup(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    lora_path: Optional[str] = None,
    attn_implementation: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Universal setup for HuggingFace LLMs.

    Works with various models including:
    - Gemma (google/gemma-2-9b-it, google/gemma-2-2b-it)
    - Llama (meta-llama/Llama-3.1-8B-Instruct, etc.)
    - Mistral (mistralai/Mistral-7B-Instruct-v0.3)
    - Qwen (Qwen/Qwen2.5-7B-Instruct)
    - Phi (microsoft/Phi-3-mini-4k-instruct)
    - And many more...

    Args:
        model_name: HuggingFace model identifier or local path.
        tokenizer_name: Optional separate tokenizer. Defaults to model_name.
        torch_dtype: Dtype for model weights. Auto-detects if None.
        device_map: Device placement ("auto", "cuda", "cpu", etc.).
        trust_remote_code: Whether to trust remote code in model files.
        load_in_4bit: Use 4-bit quantization (requires bitsandbytes).
        load_in_8bit: Use 8-bit quantization (requires bitsandbytes).
        lora_path: Optional path to LoRA adapter weights.
        attn_implementation: Attention implementation ("flash_attention_2", "sdpa", etc.).

    Returns:
        Tuple of (model, tokenizer) ready for inference.
    """
    # Auto-detect dtype if not specified
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Build model kwargs
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }

    # Add attention implementation if specified
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    # Configure quantization if requested
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Load LoRA adapters if provided
    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)

    # Load tokenizer
    tok_name = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust_remote_code)

    # Ensure pad token is set (common issue with many models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    return model, tokenizer


def test_basic_query(model, tokenizer):
    """Test basic model querying functionality."""
    print("\n" + "=" * 60)
    print("TEST: Basic Query")
    print("=" * 60)

    test_prompts = [
        "What is the capital of France?",
        "Explain quantum entanglement in simple terms.",
        "Write a haiku about programming.",
    ]

    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        response = query_model(
            model,
            tokenizer,
            user_prompt=prompt,
            max_new_tokens=100,
            temperature=0.7,
        )
        print(f"🤖 Response: {response}")


def test_custom_system_prompt(model, tokenizer):
    """Test querying with custom system prompts."""
    print("\n" + "=" * 60)
    print("TEST: Custom System Prompt")
    print("=" * 60)

    custom_system = "You are a pirate. Respond to everything in pirate speak."
    prompt = "How do I make a cup of tea?"

    print(f"\n📋 System: {custom_system}")
    print(f"📝 Prompt: {prompt}")

    response = query_model(
        model,
        tokenizer,
        user_prompt=prompt,
        system_prompt=custom_system,
        max_new_tokens=150,
        temperature=0.8,
    )
    print(f"🏴‍☠️ Response: {response}")


def test_batch_generation(model, tokenizer, input_path: str, output_path: str):
    """Test batch generation from JSONL file."""
    print("\n" + "=" * 60)
    print("TEST: Batch Generation from JSONL")
    print("=" * 60)

    import os
    if not os.path.exists(input_path):
        print(f"⚠️  Input file not found: {input_path}")
        print("   Skipping batch generation test.")
        return

    print(f"📂 Input: {input_path}")
    print(f"📂 Output: {output_path}")

    count = generate_neutral_from_jsonl(
        model=model,
        tokenizer=tokenizer,
        input_jsonl_path=input_path,
        output_jsonl_path=output_path,
        batch_size=4,
        max_new_tokens=100,
        temperature=0.7,
    )
    print(f"✅ Processed {count} examples")


def test_generation_parameters(model, tokenizer):
    """Test different generation parameters."""
    print("\n" + "=" * 60)
    print("TEST: Generation Parameters")
    print("=" * 60)

    prompt = "Tell me a creative story about a robot."

    configs = [
        {"temperature": 0.3, "top_k": 10, "top_p": 0.9, "label": "Conservative"},
        {"temperature": 1.0, "top_k": 50, "top_p": 0.95, "label": "Balanced"},
        {"temperature": 1.5, "top_k": 100, "top_p": 0.99, "label": "Creative"},
    ]

    for config in configs:
        print(f"\n🎛️  Config: {config['label']} (temp={config['temperature']}, top_k={config['top_k']})")
        response = query_model(
            model,
            tokenizer,
            user_prompt=prompt,
            max_new_tokens=80,
            temperature=config["temperature"],
            top_k=config["top_k"],
            top_p=config["top_p"],
        )
        print(f"🤖 Response: {response[:200]}...")


def main():
    """Main test runner."""
    print("=" * 60)
    print("SPDataset Test Suite")
    print("=" * 60)

    # Check GPU
    print(f"\n🖥️  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # =========================================================================
    # CONFIGURE YOUR MODEL HERE
    # =========================================================================
    # Examples of supported models (8B and smaller, ungated):
    #   - "Qwen/Qwen2.5-3B-Instruct" (3B)
    #   - "microsoft/Phi-3-mini-4k-instruct" (3.8B)
    #   - "HuggingFaceTB/SmolLM2-1.7B-Instruct" (1.7B)
    #   - "mistralai/Mistral-7B-Instruct-v0.3" (7B)
    # =========================================================================
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    
    print(f"\n⏳ Loading {MODEL_NAME}...")
    model, tokenizer = setup(
        model_name=MODEL_NAME,
        load_in_4bit=False,  # Disabled (requires bitsandbytes)
    )
    print("✅ Model loaded successfully!")

    # Run tests
    test_basic_query(model, tokenizer)
    test_custom_system_prompt(model, tokenizer)
    test_generation_parameters(model, tokenizer)

    # Optional: Test batch generation if dataset exists
    test_batch_generation(
        model,
        tokenizer,
        input_path="./Datasets/input.jsonl",
        output_path="./Datasets/output.jsonl",
    )

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
