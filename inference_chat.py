import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen1.5-1.8B-Chat"
ADAPTER = "qwen15-ww2-lora-chat"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()


def build_chat(user_message: str) -> str:
    return (
        f"<|im_start|>user\n{user_message}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def extract_assistant(text: str) -> str:
    # We expect: user ... <|im_end|>\n<|im_start|>assistant\nANSWER<|im_end|>...
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant", 1)[1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    return text.strip()


def ask(question: str) -> str:
    prompt = build_chat(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,   # greedy
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    answer = extract_assistant(decoded)
    return answer


if __name__ == "__main__":
    q1 = "What event is commonly considered the official start of World War II?"
    print("Q1:", q1)
    print("A1:", ask(q1))
    print()

    q2 = "In the context of World War II, what were the ideological goals of Nazi Germany in World War II?"
    print("Q2:", q2)
    print("A2:", ask(q2))

