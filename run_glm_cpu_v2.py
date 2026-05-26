from transformers import AutoTokenizer, AutoModel
import torch

model_name = "/mnt/data/chatglm3-6b"

prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"

print("正在加载 ChatGLM3-6B 模型...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load model with proper configuration
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="cpu",  # Explicitly set CPU
    torch_dtype=torch.float32  # Use float32 for CPU inference
).eval()

print("模型加载完成！")
print(f"\n问题: {prompt}\n")
print("回答: ")

# Use the model's chat method
response, history = model.chat(tokenizer, prompt, history=[])
print(response)

# Print model info
print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
