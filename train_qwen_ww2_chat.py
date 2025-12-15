import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen1.5-1.8B-Chat"
DATA = "data/ww2-json1000.json"
OUT_DIR = "qwen15-ww2-lora-chat"


def build_chat_prompt(messages):
    """
    Turn:
      [{"role": "user", "content": "..."},
       {"role": "assistant", "content": "..."}]
    into Qwen-style chat text.
    """
    s = ""
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "user":
            s += f"<|im_start|>user\n{content}\n<|im_end|>\n"
        elif role == "assistant":
            s += f"<|im_start|>assistant\n{content}\n<|im_end|>\n"
        elif role == "system":
            s += f"<|im_start|>system\n{content}\n<|im_end|>\n"
    return s


def main():
    # 1. Load chat-format dataset
    ds = load_dataset("json", data_files=DATA, split="train")

    def format_example(example):
        messages = example["messages"]
        text = build_chat_prompt(messages)
        return {"text": text}

    ds = ds.map(format_example, remove_columns=ds.column_names)

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=1024,
            padding=False,
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    # 3. Load base model in 4-bit (QLoRA-ish)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        load_in_4bit=True,  # deprecated warning is OK for now
        trust_remote_code=True,
    )

    # 4. LoRA config
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_cfg)

    # 5. Data collator (causal LM objective)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 6. Training args
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()

    # 8. Save LoRA adapter and tokenizer
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("Saved chat LoRA adapter to", OUT_DIR)


if __name__ == "__main__":
    main()

