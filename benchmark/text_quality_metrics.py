import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Function to calculate BLEU score
def calculate_bleu(reference, candidate):
    # Tokenize the sentences
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    
    # Calculate BLEU score
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens)
    return bleu_score

def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE score
    scores = scorer.score(reference, candidate)
    return scores

# Main function
def evaluate_text_quality():
    # Input text from the user
    reference = input("Enter the reference text: ")
    candidate = input("Enter the candidate text to evaluate: ")
    
    # Calculate BLEU score
    bleu_score = calculate_bleu(reference, candidate)
    print(f"BLEU Score: {bleu_score:.4f}")
    
    # Calculate ROUGE score
    rouge_scores = calculate_rouge(reference, candidate)
    print("ROUGE Scores:")
    print(f"ROUGE-1: {rouge_scores['rouge1']}")
    print(f"ROUGE-2: {rouge_scores['rouge2']}")
    print(f"ROUGE-L: {rouge_scores['rougeL']}")

# Run the function
evaluate_text_quality()