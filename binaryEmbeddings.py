from sklearn.random_projection import GaussianRandomProjection
from openai import OpenAI
import os
import numpy as np
import json

EMBEDDING_MODEL = "text-embedding-3-small"

class binaryEmbeddings():
    def __init__(self, openai_api_key, cacheName="embeddingCache.json"):
        global client 
        client = OpenAI(api_key = openai_api_key,)
        self.cacheName=cacheName
        self.loadCache()

    def loadCache(self):
        if os.path.exists(self.cacheName):
                with open(self.cacheName, "r") as f:
                    self.cache = json.load(f)
        else:
            self.cache = {}

    def saveCache(self):
        with open(self.cacheName, "w") as f:
            json.dump(self.cache, f, indent=4) #indent = 4 helps with formatting here

    def get_embedding(self, word, cache=True):

        if word in self.cache:
             print("found text in cache")
             return self.cache[word]
        print("calling openai")
        response = client.embeddings.create(
            input=word,
            model=EMBEDDING_MODEL
        )

        if cache:
            self.cache[word]=response.data[0].embedding
            self.saveCache()
        return response.data[0].embedding


    def embeddingToBinary(self, embedding):
        embedding = np.array(embedding).reshape(1, -1)

            # Initialize and fit LSH model
        lsh = GaussianRandomProjection(n_components=64)
        lsh.fit(embedding)
        projected_embedding = lsh.transform(embedding.reshape(1, -1)) 
        binary_embedding = (projected_embedding > 0).astype(int).flatten().tolist()  # Convert to list

        return binary_embedding
