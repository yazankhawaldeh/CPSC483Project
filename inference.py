import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen1.5-1.8B-Chat"
ADAPTER = "qwen15-ww2-lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

model = PeftModel.from_pretrained(model, ADAPTER)


def extract_answer(full_text: str) -> str:
    # We trained on: "Instruction: ...\nAnswer: <answer>"
    # 1) Keep only after the first "Answer:"
    if "Answer:" in full_text:
        full_text = full_text.split("Answer:", 1)[1]
    # 2) Cut at the next instruction-like marker
    stop_markers = ["\nInstruction:", "\nQ:", "\nQ2:"]
    cut = len(full_text)
    for m in stop_markers:
        idx = full_text.find(m)
        if idx != -1:
            cut = min(cut, idx)
    full_text = full_text[:cut]
    return full_text.strip()


def ask(instruction: str, inp: str = "") -> str:
    instr = (
    	"Give a short, exam-style factual answer. "
        "Do not introduce new questions. "
        "Do not continue with additional Q&A.\n"
        f"{instruction}"
    
    )
    if inp.strip():
        prompt = f"Instruction: {instruction}\nInput: {inp}\nAnswer:"
    else:
        prompt = f"Instruction: {instruction}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=96,    # keep it short
        do_sample=False,      # GREEDY: less hallucinations
        temperature=0.1,      # ignored when do_sample=False
        top_p=0.9,
    )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = extract_answer(decoded)
    return answer


if __name__ == "__main__":
    q1 = "Was Germany an Allied or Axis country?"
    a1 = ask(q1)
    print("Q:", q1)
    print("A:", a1)
    print()

    q2 = "Which country were the Nazi's from?"
    a2 = ask(q2)
    print("\nQ2:", q2)
    print("A2:", a2)

