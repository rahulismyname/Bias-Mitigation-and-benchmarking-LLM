from transformers import T5Tokenizer, T5ForConditionalGeneration

def use_finetuned_model(model_path, input_text):
    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    # Preprocess the input text
    input_text = f"Debias: {input_text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the debiased text
    outputs = model.generate(
        inputs.input_ids,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    # Decode the output tokens
    debiased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return debiased_text

if __name__ == "__main__":
    # Path to the fine-tuned model directory
    model_path_for_safellm = "./t5-finetuned-instruction-safe"

    model_path_for_debias = "./t5-debias-finetuned"

    # Example biased input
    # biased_text = "Get this through your head! I am tired of explaining myself over and over again."
    biased_text = "All mothers are homemakers"
    # Use the model to generate the debiased text
    debiased_output = use_finetuned_model(model_path_for_debias, biased_text)

    # Print the result
    print("Biased Text:", biased_text)
    print("Debiased Text from debias:", debiased_output)
    print("Debiased Text from safellm:", use_finetuned_model(model_path_for_safellm, biased_text))
