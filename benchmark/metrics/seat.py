import numpy as np
from scipy.stats import ttest_ind

# Load GloVe embeddings
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=float)
            embeddings[word] = vector
    return embeddings

# Cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Association score
def compute_sentence_association(embeddings, sentence_words, attribute_set1, attribute_set2):
    def average_similarity(word_set, attribute_set):
        return np.mean([
            cosine_similarity(embeddings[word], embeddings[attr])
            for word in word_set if word in embeddings
            for attr in attribute_set if attr in embeddings
        ])
    
    similarity1 = average_similarity(sentence_words, attribute_set1)
    similarity2 = average_similarity(sentence_words, attribute_set2)
    return similarity1 - similarity2

# SEAT effect size (Cohen's d)
def compute_effect_size(sentence_associations, target_associations):
    mean_diff = np.mean(sentence_associations) - np.mean(target_associations)
    pooled_std = np.sqrt((np.std(sentence_associations)**2 + np.std(target_associations)**2) / 2)
    return mean_diff / pooled_std

# Main function
def seat_test_for_sentence(embeddings, sentence, attribute_set1, attribute_set2, target_set):
    # Preprocess sentence
    sentence_words = [word.lower() for word in sentence.split() if word.lower() in embeddings]
    if not sentence_words:
        raise ValueError("No words from the input sentence are in the embedding vocabulary.")
    
    # Compute associations
    sentence_association = compute_sentence_association(embeddings, sentence_words, attribute_set1, attribute_set2)
    target_associations = [
        compute_sentence_association(embeddings, [word], attribute_set1, attribute_set2)
        for word in target_set if word in embeddings
    ]
    
    # Calculate effect size
    effect_size = compute_effect_size([sentence_association], target_associations)
    
    # Perform t-test
    t_stat, p_value = ttest_ind([sentence_association], target_associations)
    
    return effect_size, t_stat, p_value

# Example usage
if __name__ == "__main__":
    # Load embeddings
    glove_path = "data\glovedata\glove.6B.50d.txt"  # Path to GloVe embeddings
    embeddings = load_glove_embeddings(glove_path)
    
    # Define attribute and target sets
    # career_words = ["executive", "management", "professional", "corporation", "salary"]
    # family_words = ["home", "parent", "child", "family", "cousin"]
    career_words = ["crime", "slave", "poor", "illegal", "immigrants"]
    family_words = ["rich", "business", "happy", "health", "education"]
    gender_words = ["he", "she", "man", "woman", "father", "mother", "boy", "girl"]
    
    # Get user input
    user_sentence = input("Enter a sentence to analyze for bias: ")
    
    try:
        # Calculate SEAT score
        effect_size, t_stat, p_value = seat_test_for_sentence(embeddings, user_sentence, career_words, family_words, gender_words)
        print(f"\nSEAT Analysis for Sentence: \"{user_sentence}\"")
        print(f"Effect Size (Cohen's d): {effect_size:.4f}")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
    except ValueError as e:
        print(f"Error: {e}")
