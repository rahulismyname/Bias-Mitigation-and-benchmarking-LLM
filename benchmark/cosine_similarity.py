import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Calculate_cosine_similarity:
    def __init__(self):
        """
        Initializes the CosineSimilarityCalculator with a CountVectorizer instance.
        """
        self.vectorizer = CountVectorizer()

    def calculate_similarity(self, sentence1: str, sentence2: str) -> float:
        """
        Calculate cosine similarity between two sentences using Bag-of-Words representation.
        
        Parameters:
            sentence1 (str): First sentence provided by the user.
            sentence2 (str): Second sentence provided by the user.
        
        Returns:
            float: Cosine similarity score between the two sentences.
        """
        # Combine the sentences and fit the vectorizer
        vectors = self.vectorizer.fit_transform([sentence1, sentence2])

        # Compute the cosine similarity between the two vectors
        similarity = cosine_similarity(vectors[0], vectors[1])

        # Converting the similarity score to %
        return f"{similarity[0][0] * 100:.2f}%"