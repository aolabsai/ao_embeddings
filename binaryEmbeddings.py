from sklearn.random_projection import GaussianRandomProjection
from openai import OpenAI

import numpy as np

EMBEDDING_MODEL = "text-embedding-3-small"

class binaryEmbeddings():
    def __init__(self, openai_api_key):
        global client 
        client = OpenAI(api_key = openai_api_key,)
        #return client
    EMBEDDING_MODEL = "text-embedding-3-small"

    def get_embedding(self, word):
        print("calling openai")
        response = client.embeddings.create(
            input=word,
            model=EMBEDDING_MODEL
        )

        return response.data[0].embedding


    def embeddingToBinary(self, embedding):
        embedding = np.array(embedding).reshape(1, -1)

            # Initialize and fit LSH model
        lsh = GaussianRandomProjection(n_components=64)
        lsh.fit(embedding)
        projected_embedding = lsh.transform(embedding.reshape(1, -1)) 
        binary_embedding = (projected_embedding > 0).astype(int).flatten().tolist()  # Convert to list

        return binary_embedding
