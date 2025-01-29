"""Embedding data file"""

from sentence_transformers import SentenceTransformer
from chunking import split_text


# Embedding text
def embedding_text(text_data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_data)
    print("Embeddings:\n", embeddings)

text_embedding = embedding_text(split_text)