import torch
import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login
HF_MODEL_REPO = ""
torch.mps.empty_cache()

login(token="")
# Load dataset from file
def load_csv_dataset(dataset_path):
    df = pd.read_csv(dataset_path, sep='\t', lineterminator='\n', on_bad_lines='skip', names=["id", "src_tok", "tgt_tok", "src_raw", "tgt_raw", "src_POS_tags", "tgt_parse_tags"])
    df = df.dropna()  # Remove any rows with missing values
    
    # Format the dataset with the required prompt
    df["input_text"] = "Debias: " + df["src_raw"]
    df["target_text"] = df["tgt_raw"]
    
    return Dataset.from_pandas(df[["input_text", "target_text"]])

# Tokenize the dataset
def tokenize_data(dataset, tokenizer, max_input_length=512, max_target_length=512):
    def preprocess_function(examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length").input_ids
        labels = [[label if label != tokenizer.pad_token_id else -100 for label in target] for target in labels]
        model_inputs["labels"] = labels
        return model_inputs

    return dataset.map(preprocess_function, batched=True)

# Specify CSV file path (Update the path as needed)
dataset_path = "./WNC/biased.full"  # Path towards the dataset

# Load dataset from CSV
dataset = load_csv_dataset(dataset_path)

# Split into training and validation sets (80-20 split)
split_dataset = dataset.train_test_split(test_size=0.2)
train_data = split_dataset["train"]
val_data = split_dataset["test"]
# Load pre-trained T5-large model and tokenizer
model_name = "deutsche-welle/t5_large_peft_wnc_debiaser"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device("mps")
model.to(device=device)
# Apply LoRA adapters for efficient fine-tuning
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,  # Low-rank dimension
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"]  # Applies LoRA to key attention modules
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Display trainable parameters

# Tokenize datasets
train_data = tokenize_data(train_data, tokenizer)
val_data = tokenize_data(val_data, tokenizer)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./finetunedmodels/t5-neutral-lora-finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    num_train_epochs=2,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=3,
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU supports it
    group_by_length=True,  # Dynamic batching for efficiency
    push_to_hub=True,
    hub_model_id=HF_MODEL_REPO,
    hub_strategy="every_save",
)
# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
)
# Fine-tune the model
trainer.train()

# Save the final fine-tuned model
model.save_pretrained("./finetunedmodels/t5-neutral-lora-finetuned")
tokenizer.save_pretrained("./finetunedmodels/t5-neutral-lora-finetuned")

print("Fine-tuned model with LoRA saved successfully!")