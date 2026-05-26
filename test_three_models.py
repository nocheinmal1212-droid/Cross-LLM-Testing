import json, torch, gc, os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# ---------- Test Questions ----------
questions = [
    {"id":1, "type":"linguistic", "prompt":"请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"},
    {"id":2, "type":"linguistic", "prompt":"请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上"},
    {"id":3, "type":"logical",   "prompt":"他知道我知道你知道他不知道吗？这句话里，到底谁不知道"},
    {"id":4, "type":"math",       "prompt":"Janet’s ducks lay 16 eggs per day. She eats three for breakfast and bakes four into muffins. She sells the rest at $2 each. How much does she earn daily?"},
    {"id":5, "type":"code",       "prompt":"Write a Python function has_close_elements(numbers, threshold) that returns True if any two numbers in the list are within threshold of each other. Include a docstring and an example."},
    {"id":6, "type":"commonsense","prompt":"Why do we wear a coat in winter?"}
]

# ---------- Model definitions (UPDATED PATHS & MEMORY OPTIMISATIONS) ----------
models_to_test = [
    {
        "name": "ChatGLM3-6B",
        "path": "/mnt/data/chatglm3-6b",
        "type": "chatglm"
    },
    {
        "name": "Qwen-7B-Chat",
        "path": "/mnt/data/Qwen-7B-Chat/qwen/Qwen-7B-Chat",
        "type": "qwen"
    },
    {
        "name": "Baichuan2-7B-Base",
        "path": "/mnt/data/Baichuan2-7B-Base",
        "type": "baichuan"
    }
]

all_results = {}

for model_info in models_to_test:
    model_name = model_info["name"]
    model_path = model_info["path"]

    if not os.path.exists(model_path):
        print(f"❌ {model_name} not found at {model_path}, skipping.")
        continue

    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load with float16 & low_cpu_mem_usage to avoid OOM
    if model_info["type"] == "chatglm":
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu"
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu"
        ).eval()

    results = []
    for q in questions:
        print(f"  Q{q['id']}: {q['prompt'][:50]}...")
        try:
            if model_info["type"] in ("chatglm", "qwen"):
                # Both ChatGLM3 and Qwen have .chat()
                response, _ = model.chat(tokenizer, q["prompt"], history=[])
            else:
                # Baichuan Base – use <human>/<bot> template
                prompt_text = f"<human>{q['prompt']}\n<bot>"
                inputs = tokenizer(prompt_text, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.8,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                    )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if response.startswith(prompt_text):
                    response = response[len(prompt_text):].strip()
        except Exception as e:
            response = f"[ERROR: {e}]"

        print(f"    → {response[:120]}...")
        results.append({
            "question_id": q["id"],
            "type": q["type"],
            "prompt": q["prompt"],
            "response": response
        })

    # Save per-model results
    out_file = f"/mnt/workspace/results_{model_name.replace('/', '_')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    all_results[model_name] = results

    # Free memory before next model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# Save combined results
with open("/mnt/workspace/all_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print("\n✅ All done. Results saved in /mnt/workspace/")
