from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch
import os
from huggingface_hub import login
from accelerate.test_utils.testing import get_backend
from dotenv import load_dotenv


def preprocess_data(examples, tokenizer, max_input_length=512, max_target_length=128):
    # Use "Original Sentence" as the input and "Debiased" as the target
    inputs = examples["Original Sentence"]
    targets = examples["Debiased"]
    print('type of debiased:', type(targets))
    # Prepend task-specific instruction
    inputs = [f"Debias: {inp}" for inp in inputs]
    print('type of inputs:', type(inputs))

    # Filter out invalid inputs or targets
    filtered_inputs, filtered_targets = [], []
    for inp, tgt in zip(inputs, targets):
        if isinstance(inp, str) and isinstance(tgt, str):
            filtered_inputs.append(f"Debias: {inp}")
            filtered_targets.append(tgt)

     # Tokenize inputs and targets
    model_inputs = tokenizer(
        filtered_inputs, max_length=max_input_length, padding="max_length", truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            filtered_targets, max_length=max_target_length, padding="max_length", truncation=True
        )

    # Replace padding token IDs in labels with -100 to ignore during loss calculation
    model_inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_ids]
        for label_ids in labels["input_ids"]
    ]

    return model_inputs


def main():
    # Dataset and columns
    dataset_name = "newsmediabias/instruction-safe-llm"
    input_column = "Original Sentence"
    target_column = "Debiased"

    # Load the dataset
    dataset = load_dataset(dataset_name)
    dataset = dataset["test"]  # Only one split named "test"
    dataset = dataset.rename_column('Debiased ','Debiased')
    # Ensure dataset has the required columns
    required_columns = [input_column, target_column]
    for column in dataset.column_names:
        if column not in required_columns:
            dataset = dataset.remove_columns(column)

    for column in required_columns:
        if column not in dataset.column_names:
            raise ValueError(f"Missing required column: {column}")
    device = torch.device("mps")

    # Load pre-trained T5 tokenizer and model
    model_name = "t5-small"  # Use a larger variant if required (e.g., 't5-base')
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device=device)
    # Preprocess the dataset
    tokenized_dataset = dataset.map(
        lambda x: preprocess_data(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Split into training and testing sets (optional, since the split is "test")
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./t5-finetuned-instruction-safe",  # Directory to save the model
        evaluation_strategy="epoch",                # Evaluate at the end of each epoch
        learning_rate=5e-5,                          # Learning rate
        per_device_train_batch_size=8,               # Batch size for training
        per_device_eval_batch_size=8,                # Batch size for evaluation
        num_train_epochs=3,                          # Number of epochs
        weight_decay=0.01,                           # Weight decay for regularization
        save_total_limit=3,                          # Limit the number of saved checkpoints
        save_strategy="epoch",                       # Save at the end of each epoch
        fp16=torch.cuda.is_available(),              # Enable FP16 if a GPU is available
        logging_dir="./logs",                        # Directory for logging
        logging_steps=100,                           # Log every 100 steps
    )

    # Data collator for dynamic padding
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

    # Save the final model
    trainer.save_model("./t5-finetuned-instruction-safe")
    print("Fine-tuning completed. Model saved to './t5-finetuned-instruction-safe'.")

if __name__ == "__main__":
    load_dotenv()
    token = os.getenv('RAHUL_TOKEN')
    login(token=token)
    main()
