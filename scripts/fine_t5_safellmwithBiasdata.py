from datasets import load_dataset, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
import os
from huggingface_hub import login
from accelerate.test_utils.testing import get_backend

# Load the dataset
def load_and_filter_dataset():
    dataset = load_dataset("newsmediabias/instruction-safe-llm")
    dataset = dataset.rename_column('Debiased ','Debiased')

    # Filter rows where BIAS == "Yes"
    filtered_data = dataset["test"].filter(lambda x: x["BIAS"] == "Yes")
    
    # Rename columns for clarity
    def format_data(row):
        return {
            "input_text": f"Debias this text: {row['Original Sentence']}",
            "target_text": row["Debiased"]
        }

    formatted_dataset = filtered_data.map(format_data, remove_columns=filtered_data.column_names)
    return formatted_dataset

# Tokenize the dataset
def tokenize_data(dataset, tokenizer, max_input_length=512, max_target_length=512):
    def preprocess_function(examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        filtered_inputs, filtered_targets = [], []
        for inp, tgt in zip(inputs, targets):
            if isinstance(inp, str) and isinstance(tgt, str):
                filtered_inputs.append(inp)
                filtered_targets.append(tgt)
            else:
                filtered_inputs.append(str(inp))
                filtered_targets.append(str(tgt))
                print("Not a string input:",inp)
                print("Not a string target:",tgt)


        model_inputs = tokenizer(filtered_inputs, max_length=max_input_length, truncation=True, padding="max_length")
        labels = tokenizer(filtered_targets, max_length=max_target_length, truncation=True, padding="max_length").input_ids
        labels = [
            [(label if label != tokenizer.pad_token_id else -100) for label in target] for target in labels
        ]
        model_inputs["labels"] = labels
        return model_inputs

    return dataset.map(preprocess_function, batched=True)

# Main function
def main():
    # Load and prepare dataset
    dataset = load_and_filter_dataset()

    # Split into training and validation sets
    split_dataset = dataset.train_test_split(test_size=0.2)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]

    # Load pre-trained T5 model and tokenizer
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    device = torch.device("mps") # GPS in Mac
    model.to(device=device)

    # Tokenize datasets
    train_data = tokenize_data(train_data, tokenizer)
    val_data = tokenize_data(val_data, tokenizer)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./t5-debias-finetune",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        fp16=False,  # Use mixed precision if GPU supports it
        push_to_hub=False  # Set to True if you want to upload the model to Hugging Face Hub
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

    # Save the final model and tokenizer
    model_dir = "./t5-debias-finetuned"
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print("Model fine-tuned and saved successfully!")
    # {'eval_loss': 2.206937551498413, 'eval_runtime': 27.3408, 'eval_samples_per_second': 35.076, 'eval_steps_per_second': 4.389, 'epoch': 3.0}                                                                                                                        
    # {'train_runtime': 2953.6953, 'train_samples_per_second': 3.896, 'train_steps_per_second': 0.488, 'train_loss': 2.5404275986883373, 'epoch': 3.0} 

if __name__ == "__main__":
    load_dotenv()
    token = os.getenv('RAHUL_TOKEN')
    login(token=token)
    main()
