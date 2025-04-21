
# Import necessary libraries
import os
import google.generativeai as genai
from typing import List, Tuple
from numpy import dot
from numpy.linalg import norm

# ------------------------------------------------------------------------------
# API Key Configuration
# ------------------------------------------------------------------------------

# Get the API key from the environment variable
api_key = os.environ.get("GEMINI_API_KEY")

# Check if the API key is set
if not api_key:
    raise EnvironmentError("Please set the GEMINI_API_KEY environment variable.")

# Configure the API
genai.configure(api_key=api_key)

# ------------------------------------------------------------------------------
# Define Word Sets
# ------------------------------------------------------------------------------

related_words: List[Tuple[str, str]] = [
    ("king", "queen"),
    ("man", "woman"),
    ("happy", "joyful"),
    ("big", "large"),
]

unrelated_words: List[Tuple[str, str]] = [
    ("king", "table"),
    ("sun", "queen"),
    ("happy", "car"),
    ("big", "small"),
]

# ------------------------------------------------------------------------------
# Embedding Generation
# ------------------------------------------------------------------------------
def get_embedding(text: str):
    """
    Generates an embedding for a given text using the Gemini API.

    Args:
        text: The input text (word or phrase).

    Returns:
        A list of floats representing the embedding, or None if an error occurred.
    """
    try:
        model = genai.GenerativeModel('models/embedding-001')
        embedding = model.embed_content(text).embedding.values
        return embedding
    except Exception as e:
        print(f"Error generating embedding for '{text}': {e}")
        return None

# ------------------------------------------------------------------------------
# Cosine Similarity Calculation
# ------------------------------------------------------------------------------

def cosine_similarity(embedding1: list, embedding2: list) -> float:
    """
    Calculates the cosine similarity between two embeddings.

    Args:
        embedding1: The first embedding (list of floats).
        embedding2: The second embedding (list of floats).

    Returns:
        The cosine similarity score (float), or None if an error occurred.
    """
    try:
        # Ensure embeddings are not None and have the same length
        if embedding1 is None or embedding2 is None or len(embedding1) != len(embedding2):
            return None

        return dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return None

# ------------------------------------------------------------------------------
# Calculate and Print Similarity Scores
# ------------------------------------------------------------------------------

print("Calculating similarity scores...")

def calculate_and_print_similarity(word_pairs: List[Tuple[str, str]], category: str):
    """
    Calculates and prints the similarity scores for a list of word pairs.

    Args:
        word_pairs: A list of tuples, where each tuple contains a pair of words.
        category: A string indicating the category of the word pairs (e.g., "Related", "Unrelated").
    """
    print(f"\n{category} Words:")
    for word1, word2 in word_pairs:
        embedding1 = get_embedding(word1)
        embedding2 = get_embedding(word2)
        if embedding1 and embedding2:  # Check if embeddings were generated successfully
            similarity_score = cosine_similarity(embedding1, embedding2)
            if similarity_score is not None:  # Check if similarity calculation was successful
                print(f"  '{word1}' and '{word2}': {similarity_score:.4f}")
            else:
                print(f"  Could not calculate similarity for '{word1}' and '{word2}'.")
        else:
            print(f"  Could not generate embeddings for '{word1}' and/or '{word2}'.")


calculate_and_print_similarity(related_words, "Related")
calculate_and_print_similarity(unrelated_words, "Unrelated")
