import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

class Text_quality_evaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

        # Apply smoothing for better BLEU scoring in edge cases
        self.smoothing = SmoothingFunction().method1

    # Function to calculate BLEU score
    def calculate_bleu(self, sentence1: str, sentence2: str):
        
        # Tokenize the sentences
        sentence1_tokens = sentence1.split()
        sentence2_tokens = sentence2.split()

        bleu_score = sentence_bleu([sentence1_tokens], sentence2_tokens, smoothing_function=self.smoothing)
        return bleu_score

    # Function to calculate ROUGE scores
    def calculate_rouge(self, sentence1: str, sentence2: str):
        scores = self.scorer.score(sentence1, sentence2)
        return scores

    # Function to interpret scores as quality preservation
    def interpret_quality(self, score):
        if score >= 0.75:
            return "Excellent"
        elif score >= 0.5:
            return "Good"
        elif score >= 0.25:
            return "Fair"
        else:
            return "Poor"
    
    # Function to evaluate the text quality
    def evaluate_text_quality(self, sentence1: str, sentence2: str):

        # output variable containing rouge and blue scores
        text_quality = dict()
        bleu_score = self.calculate_bleu(sentence1, sentence2)
        bleu_quality = self.interpret_quality(bleu_score)
        rouge_scores = self.calculate_rouge(sentence1, sentence2)
        text_quality['BLEU Score'] = f"{bleu_score:.4f} (Quality: {bleu_quality})"
        text_quality['Rouge'] = f"ROUGE-1: F1 Score: {rouge_scores['rouge1'].fmeasure:.4f} ROUGE-N: F1 Score: {rouge_scores['rouge2'].fmeasure:.4f}"

        return text_quality