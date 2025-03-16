import numpy as np
from scipy.spatial.distance import cosine
import gensim.downloader as api
from spacy.lang.en import English

class Weatscore:
    def __init__(self):
        self.word_vectors = api.load("word2vec-google-news-300")
        self.nlp = English()
        self.tokenizer = self.nlp.tokenizer
        self.attribute_words_bias = [
            "exploitation", "misuse", "oppression", "injustice", "magnificient", "dominant", "exposed", "scandal", "desperately", "proves",
            "corrupt", "manipulated", "incompetent", "ruined", "Dishonest", "misled", "stubborn", "Extremist", "radical", "unethical", "deceived",
            "unfortunately", "heresy", "men", "odious", "speculative", "worst", "famously", "controversy", "unfortunate", "classic", "legendary",
            "massive", "prestigious", "sophisticated", "most", "claimed", "popular", "acclaimed", "adopts", "heretical", "comedienne", "premier", "rare",
            "simple", "persian", "illegally", "strongest", "crown", "jewel", "successful", "admirably", "so", "called", "passed", "centered",
            "inspirational", "iconic", "develop", "grotesque", "visionary", "away", "complete", "unfettered", "admitted", "ultimatum", "peace", "massacre",
            "nearly", "review", "excellent", "controversial", "upgraded"
        ]
       
        self.attribute_words_neutral = [
            "actions", "practice", "use", "policy", "described", "may", "prove", "influenced", "affected", "reported", "disputed", "acknowledge",
            "controversial", "profitability", "position", "people", "proposed", "pointed", "performing", "relatively", "talented",
            "enigmatic", "improved", "significant", "well", "died", "focused", "comedian", "propose", "facial", "evidence",
            "full", "acquiescence", "killings", "almost", "critics", "changed"
        ]

    def calculate_similarity(self, word, attribute_words):
        """Calculate cosine similarities for a word against a list of attribute words."""
        similarities = []
        for attr_word in attribute_words:
            try:
                similarities.append(1 - cosine(self.word_vectors[word], self.word_vectors[attr_word]))
            except KeyError:
                print(f"Word {word} or {attr_word} not found in the vocabulary.")
        return np.mean(similarities) if similarities else None

     # Process and clean sentences
    def tokenize_sentence(self, sentence):
        """Tokenizes a sentence into words."""
        return [token.text.lower() for token in self.tokenizer(sentence) if token.is_alpha]

    def calculate_weat_score(self, sentence):
        """
        Calculate WEAT score for a given sentence.
        sentence: Input Sentence.
        """
        scores_1 = []
        scores_2 = []
        sentence_token = self.tokenize_sentence(sentence)
        for word in sentence_token:
            if word in self.word_vectors:
                score_1 = self.calculate_similarity(word, self.attribute_words_bias)
                score_2 = self.calculate_similarity(word, self.attribute_words_neutral)
                if score_1 is not None and score_2 is not None:
                    scores_1.append(score_1)
                    scores_2.append(score_2)

        if scores_1 and scores_2:
            # Compute WEAT score
            return f"{np.mean(scores_1) - np.mean(scores_2):.4f}"
        else:
            return "Unable to compute WEAT score (missing embeddings for tokens)."