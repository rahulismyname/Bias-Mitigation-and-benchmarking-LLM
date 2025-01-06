import numpy as np
from transformers import pipeline
from scipy.spatial.distance import cosine
import gensim.downloader as api

# Initialize zero-shot classifier
classifier = pipeline("zero-shot-classification")

# Load word2vec embeddings model (you can also use other embedding models)
word_vectors = api.load("word2vec-google-news-300")

# Define labels (for detecting bias)
labels = ['biased', 'unbiased']

def detect_bias(sentence):
    # Use zero-shot classification to check if the sentence is biased or unbiased
    result = classifier(sentence, candidate_labels=labels)
    label = result['labels'][0]
    return label

def detect_bias(sentence):
    # Use zero-shot classification to check if the sentence is biased or unbiased
    result = classifier(sentence, candidate_labels=labels)
    label = result['labels'][0]
    return label

def weat_score(target_words, attribute_words_1, attribute_words_2):
    # Calculate the cosine similarity for the word sets
    similarities_1 = []
    similarities_2 = []

    for word1 in target_words:
        for word2 in attribute_words_1:
            similarities_1.append(1 - cosine(word_vectors[word1], word_vectors[word2]))  # Cosine similarity between word1 and word2 in attribute group 1
        for word2 in attribute_words_2:
            similarities_2.append(1 - cosine(word_vectors[word1], word_vectors[word2]))  # Cosine similarity between word1 and word2 in attribute group 2

    # Compute the WEAT score
    weat = (np.mean(similarities_1) - np.mean(similarities_2)) / (
        len(attribute_words_1) * len(target_words) + len(attribute_words_2) * len(target_words)
    )
    return weat

# Example words for bias detection
gender_words = ['man', 'woman', 'men', 'women', 'male', 'female', 'girl', 'boy', 'mother', 'father', 'husband', 'wife', 'transgender']
age_words = ['young', 'old', 'elderly', 'teenager', 'senior', 'older', 'older workers', 'young people', 'younger', 'elder']
ethnicity_words = ['asian', 'black', 'immigrant', 'native', 'foreign', 'rural', 'urban', 'foreign workers', 'people from rural areas', 'people from urban areas']
health_words = ['obese', 'diabetics', 'addicts', 'cancer patients', 'mentally ill', 'disabled', 'pregnant women', 'patients with tattoos']
economic_status_words = ['poor', 'rich', 'high-net-worth', 'wealthy', 'low-income', 'poor people', 'high-income clients']
education_words = ['graduates', 'engineers', 'college degrees', 'university', 'students', 'teachers', 'professors', 'interns']
profession_words = ['nurses', 'doctors', 'scientists', 'teachers', 'athletes', 'workers', 'employees', 'staff', 'supervisors', 'managers', 'employees from rural areas', 'employees from urban areas']
social_role_words = ['housewives', 'parents', 'leaders', 'supervisors', 'principals', 'teachers', 'managers', 'employees', 'workers']
ability_words = ['better', 'worse', 'good at', 'bad at', 'strong', 'weak', 'natural', 'unqualified', 'capable', 'incapable']
characteristics_words = ['logical', 'emotional', 'aggressive', 'soft', 'cautious', 'confident', 'nurturing', 'empathetic', 'harsh', 'unreliable', 'reliable']
behavior_words = ['lazy', 'careless', 'serious', 'dedicated', 'focused', 'distracted', 'innovative', 'uninnovative', 'reliable', 'unreliable', 'committed', 'uncommitted', 'strict', 'lenient']
generalizations_words = ['everyone', 'no one', 'always', 'never', 'all', 'most', 'some', 'none', 'rarely', 'often', 'never', 'usually', 'only', 'just', 'people always', 'only', 'all people']
good_words = ['intelligent', 'capable', 'talented', 'successful', 'honest']
bad_words = ['ignorant', 'incompetent', 'unsuccessful', 'dishonest', 'lazy']


# Example usage:
sentence = input("Enter a sentence to check its bias and calculate WEAT score: ")
print(sentence)

# Check if the sentence is biased
bias_status = detect_bias(sentence)

if bias_status == 'biased':
    print("The sentence is biased.")
    # Calculate the WEAT score (you can replace words with more appropriate ones based on the context)
    score = weat_score(gender_words, good_words, bad_words)
    print(f"WEAT Score: {score}")
else:
    print("The sentence is unbiased.")