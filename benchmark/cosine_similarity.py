import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate cosine similarity between two sentences using Bag-of-Words representation.
    
    Parameters:
        sentence1 (str): First sentence provided by the user.
        sentence2 (str): Second sentence provided by the user.
        
    Returns:
        float: Cosine similarity score between the two sentences.
    """
    # Create a CountVectorizer instance to represent the sentences as vectors
    vectorizer = CountVectorizer()
    
    # Combine the sentences and fit the vectorizer
    vectors = vectorizer.fit_transform([sentence1, sentence2])
    
    # Compute the cosine similarity between the two vectors
    similarity = cosine_similarity(vectors[0], vectors[1])
    
    return similarity[0][0]

# User input for two sentences
sentence1 = input("Enter the first sentence: ")
sentence2 = input("Enter the second sentence: ")

# Calculate cosine similarity
similarity_score = calculate_cosine_similarity(sentence1, sentence2)

# Calculate match percentage
match_percentage = similarity_score * 100

# Display the similarity score and match percentage
print(f"Cosine Similarity: {similarity_score:.4f}")
print(f"The sentences match by approximately {match_percentage:.2f}%.")
