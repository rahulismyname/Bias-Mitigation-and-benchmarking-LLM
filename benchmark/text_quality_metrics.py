import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Function to calculate BLEU score
def calculate_bleu(reference, candidate):
    # Tokenize the sentences
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()

    # Apply smoothing for better BLEU scoring in edge cases
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)
    return bleu_score

# Function to calculate ROUGE scores
def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

# Function to interpret scores as quality preservation
def interpret_quality(score):
    if score >= 0.75:
        return "Excellent"
    elif score >= 0.5:
        return "Good"
    elif score >= 0.25:
        return "Fair"
    else:
        return "Poor"

# Main function
def evaluate_text_quality():
    # Input text from the user
    reference = input("Enter the reference text: ")
    candidate = input("Enter the candidate text to evaluate: ")

    # Calculate BLEU score
    bleu_score = calculate_bleu(reference, candidate)
    bleu_quality = interpret_quality(bleu_score)
    print(f"BLEU Score: {bleu_score:.4f} (Quality: {bleu_quality})")

    # Calculate ROUGE scores
    rouge_scores = calculate_rouge(reference, candidate)

    print("ROUGE Scores:")
    print(f"ROUGE-1: F1 Score: {rouge_scores['rouge1'].fmeasure:.4f} ")
    print(f"ROUGE-2: F1 Score: {rouge_scores['rouge2'].fmeasure:.4f} ")
    print(f"ROUGE-L: F1 Score: {rouge_scores['rougeL'].fmeasure:.4f} ")

# Run the function
evaluate_text_quality()
