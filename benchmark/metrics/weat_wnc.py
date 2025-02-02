import numpy as np
from scipy.spatial.distance import cosine
import gensim.downloader as api
from spacy.lang.en import English

# Load required models
print("Loading Word2Vec model...")
word_vectors = api.load("word2vec-google-news-300")
print("Word2Vec model loaded successfully!")

# Tokenizer using SpaCy
nlp = English()
tokenizer = nlp.tokenizer

def calculate_similarity(word, attribute_words):
    """Calculate cosine similarities for a word against a list of attribute words."""
    similarities = []
    for attr_word in attribute_words:
        try:
            similarities.append(1 - cosine(word_vectors[word], word_vectors[attr_word]))
        except KeyError:
            print(f"Word {word} or {attr_word} not found in the vocabulary.")
    return np.mean(similarities) if similarities else None

def calculate_weat_score(sentence_tokens, attribute_words_1, attribute_words_2):
    """
    Calculate WEAT score for a given sentence.
    sentence_tokens: List of tokenized words in the sentence.
    attribute_words_1: First set of attribute words (biased context).
    attribute_words_2: Second set of attribute words (neutral context).
    """
    scores_1 = []
    scores_2 = []

    for word in sentence_tokens:
        if word in word_vectors:
            score_1 = calculate_similarity(word, attribute_words_1)
            score_2 = calculate_similarity(word, attribute_words_2)
            if score_1 is not None and score_2 is not None:
                scores_1.append(score_1)
                scores_2.append(score_2)

    if scores_1 and scores_2:
        # Compute WEAT score
        return np.mean(scores_1) - np.mean(scores_2)
    else:
        return None

# Process and clean sentences
def tokenize_sentence(sentence):
    """Tokenizes a sentence into words."""
    return [token.text.lower() for token in tokenizer(sentence) if token.is_alpha]

# Define biased and neutral attribute word groups
attribute_words_bias = ["exploitation", "misuse", "oppression", "injustice", "magnificient", "dominant", "exposed","scandal","desperately","proves","corrupt","manipulated","incompetent", "ruined","Dishonest","misled","stubborn","Extremist","radical","unethical","deceived"]
attribute_words_neutral = ["actions", "practice", "use", "policy", "described","may" ,"prove","influenced", "affected","reported", "disputed","acknowledge","controversial","profitability"]

# Get input for the biased and neutralized cases
biased_sentence = input("Enter the biased sentence: ")
neutralized_sentence = input("Enter the neutralized sentence: ")

# Store examples dynamically
examples = {
    "Biased (Source)": biased_sentence,
    "Neutralized (Target)": neutralized_sentence
}

# Calculate WEAT scores
print("\nCalculating WEAT scores for each input sentence...\n")
for example_type, sentence in examples.items():
    tokens = tokenize_sentence(sentence)
    weat_score = calculate_weat_score(tokens, attribute_words_bias, attribute_words_neutral)
    if weat_score is not None:
        print(f"{example_type} - WEAT Score: {weat_score:.4f}")
    else:
        print(f"{example_type} - Unable to compute WEAT score (missing embeddings for tokens).")