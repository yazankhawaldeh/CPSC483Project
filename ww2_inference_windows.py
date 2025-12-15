import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen1.5-1.8B-Chat"


ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "qwen15-ww2-lora-chat")

USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
DTYPE = torch.float16 if USE_CUDA else torch.float32


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    dtype=DTYPE,  
)

model = model.to(DEVICE)


adapter_config_path = os.path.join(ADAPTER_DIR, "adapter_config.json")
if os.path.isdir(ADAPTER_DIR) and os.path.isfile(adapter_config_path):
    print(f"[OK] Loading LoRA chat adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
else:
    print(f"[WARN] LoRA adapter not found at: {ADAPTER_DIR}")
    print("[WARN] Running BASE model only (not fine-tuned).")

model.eval()



def build_chat_prompt(messages):
    """
    Prefer tokenizer.apply_chat_template if available (Qwen chat models usually support it).
    Fallback to a simple role-labeled format if not.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


    prompt = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        prompt += f"{role.upper()}: {content}\n"
    prompt += "ASSISTANT: "
    return prompt


@torch.inference_mode()
def generate_reply(messages, max_new_tokens=256):
    prompt = build_chat_prompt(messages)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  
    )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)


    if decoded.startswith(prompt):
        reply = decoded[len(prompt):].strip()
    else:
        reply = decoded.strip()

    return reply


def main():
    print("=== WWII Qwen LoRA Chat (Windows-safe) ===")
    print("Type 'exit' to quit.\n")

    messages = [
        {"role": "system", "content": "You are a helpful assistant focused on World War II questions."}
    ]

    while True:
        user_msg = input("You: ").strip()
        if user_msg.lower() in ("exit", "quit"):
            break
        if not user_msg:
            continue

        messages.append({"role": "user", "content": user_msg})
        reply = generate_reply(messages)
        messages.append({"role": "assistant", "content": reply})

        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()
