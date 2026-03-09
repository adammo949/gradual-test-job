import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

model_name = "facebook/opt-125m"
output_dir = os.environ.get("OUTPUT_DIR", "./output")

print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Loading dataset...")
dataset = load_dataset("imdb", split="train[:200]")

def tokenize(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    max_steps=20,
    per_device_train_batch_size=4,
    save_steps=10,
    logging_steps=5,
    no_cuda=False,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=data_collator
)

print("Starting training...")
trainer.train()
trainer.save_model(output_dir)
print("Training complete. Model saved to", output_dir)
