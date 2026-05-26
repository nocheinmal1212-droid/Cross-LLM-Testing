"""
test_three_models_v2.py
Comparative evaluation of three open-source Chinese LLMs on a fixed prompt suite.
"""

import json
import torch
import gc
import os
import random
import time
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# ---------- Reproducibility ----------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ---------- Config ----------
OUTPUT_DIR = "/mnt/workspace"
PREVIEW_CHARS = 300
DTYPE = torch.bfloat16

# ---------- Test Questions ----------
questions = [
    {"id": 1, "type": "linguistic",
     "prompt": "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"},
    {"id": 2, "type": "linguistic",
     "prompt": "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上"},
    {"id": 3, "type": "logical",
     "prompt": "他知道我知道你知道他不知道吗？这句话里，到底谁不知道"},
    {"id": 4, "type": "math",
     "prompt": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast and bakes four into muffins. She sells the rest at $2 each. How much does she earn daily?"},
    {"id": 5, "type": "code",
     "prompt": "Write a Python function has_close_elements(numbers, threshold) that returns True if any two numbers in the list are within threshold of each other. Include a docstring and an example."},
    {"id": 6, "type": "commonsense",
     "prompt": "Why do we wear a coat in winter?"},
]

# ---------- Model definitions ----------
models_to_test = [
    {"name": "ChatGLM3-6B",
     "path": "/mnt/data/chatglm3-6b",
     "type": "chatglm"},
    {"name": "Qwen-7B-Chat",
     "path": "/mnt/data/Qwen-7B-Chat/qwen/Qwen-7B-Chat",
     "type": "qwen"},
    {"name": "Baichuan2-7B-Chat",
     "path": "/mnt/data/Baichuan2-7B-Chat",
     "type": "baichuan_chat"},
]


def force_greedy(model):
    """Override the model's generation_config to disable sampling."""
    gc_obj = getattr(model, "generation_config", None)
    if gc_obj is not None:
        gc_obj.do_sample = False
        gc_obj.temperature = 1.0
        gc_obj.top_p = 1.0
        # top_k can stay as-is; with do_sample=False it has no effect


# ---------- Clean stale outputs ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)
combined_path = os.path.join(OUTPUT_DIR, "all_results.json")
metadata_path = os.path.join(OUTPUT_DIR, "run_metadata.json")
for m in models_to_test:
    p = os.path.join(OUTPUT_DIR, f"results_{m['name'].replace('/', '_')}.json")
    if os.path.exists(p):
        os.remove(p)
for p in (combined_path, metadata_path):
    if os.path.exists(p):
        os.remove(p)

# ---------- Main loop ----------
all_results = {}
load_times = {}

for model_info in models_to_test:
    model_name = model_info["name"]
    model_path = model_info["path"]

    if not os.path.exists(model_path):
        print(f"❌ {model_name} not found at {model_path}, skipping.")
        continue

    print(f"\n{'='*60}\nTesting {model_name}\n{'='*60}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    loader = AutoModel if model_info["type"] == "chatglm" else AutoModelForCausalLM
    model = loader.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        device_map="cpu",
    ).eval()

    force_greedy(model)
    load_sec = time.time() - t0
    load_times[model_name] = round(load_sec, 1)
    print(f"  loaded in {load_sec:.1f}s")

    results = []
    for q in questions:
        print(f"  Q{q['id']} [{q['type']}]: {q['prompt'][:50]}...")
        t_start = time.time()
        # Reset seed per prompt so prompt order does not change outputs
        torch.manual_seed(SEED)
        try:
            if model_info["type"] == "chatglm":
                # ChatGLM3.chat accepts decoding args directly
                response, _ = model.chat(
                    tokenizer, q["prompt"], history=[],
                    do_sample=False, temperature=1.0, top_p=1.0,
                )
            elif model_info["type"] == "qwen":
                # Qwen-Chat uses generation_config; force_greedy already set it
                response, _ = model.chat(tokenizer, q["prompt"], history=None)
            elif model_info["type"] == "baichuan_chat":
                # Baichuan2-Chat takes a messages list
                messages = [{"role": "user", "content": q["prompt"]}]
                response = model.chat(tokenizer, messages)
            else:
                response = f"[ERROR: unknown type {model_info['type']}]"
        except Exception as e:
            response = f"[ERROR: {type(e).__name__}: {e}]"

        elapsed = time.time() - t_start
        truncated = response[:PREVIEW_CHARS]
        suffix = "..." if len(response) > PREVIEW_CHARS else ""
        print(f"    ({elapsed:.1f}s) → {truncated}{suffix}")

        results.append({
            "question_id": q["id"],
            "type": q["type"],
            "prompt": q["prompt"],
            "response": response,
            "elapsed_sec": round(elapsed, 2),
        })

    out_file = os.path.join(OUTPUT_DIR, f"results_{model_name.replace('/', '_')}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  saved → {out_file}")

    all_results[model_name] = results

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------- Save combined + metadata ----------
with open(combined_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

run_metadata = {
    "seed": SEED,
    "dtype": str(DTYPE).replace("torch.", ""),
    "device": "cpu",
    "decoding": "greedy (do_sample=False)",
    "preview_chars": PREVIEW_CHARS,
    "models_attempted": [m["name"] for m in models_to_test],
    "models_run": list(all_results.keys()),
    "load_times_sec": load_times,
}
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(run_metadata, f, ensure_ascii=False, indent=2)

print(f"\n✅ All done.")
print(f"   Combined: {combined_path}")
print(f"   Metadata: {metadata_path}")