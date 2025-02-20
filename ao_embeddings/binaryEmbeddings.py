from sklearn.random_projection import GaussianRandomProjection
from openai import OpenAI
import os
import numpy as np
import json

EMBEDDING_MODEL = "text-embedding-3-small"

class binaryEmbeddings():
    def __init__(self, openai_api_key, cacheName="embeddingCache.json", numberBinaryDigits=64):
        global client 
        client = OpenAI(api_key = openai_api_key,)
        self.cacheName=cacheName
        self.loadCache()
        self.lsh = GaussianRandomProjection(n_components=numberBinaryDigits, random_state=42)

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

    def get_embedding_batch(self, words, cache=True):
        keys = str(words)
        # if keys in self.cache:
        #     print("found text in cache")
        #     return self.cache[keys]
        
        response = client.embeddings.create(
            input=words,
            model=EMBEDDING_MODEL
        )
        
        # Extract all embeddings from the response
        embeddings = [item.embedding for item in response.data]
        
        # if cache:
        #     self.cache[keys] = embeddings
        #     self.saveCache()
        
        return embeddings



    def embeddingToBinary(self, embedding):
        embedding = np.array(embedding).reshape(1, -1)

            # Initialize and fit LSH model
        self.lsh.fit(embedding)
        projected_embedding = self.lsh.transform(embedding.reshape(1, -1))
        binary_embedding = (projected_embedding > 0).astype(int).flatten().tolist()  # Convert to list

        return binary_embedding
    
    def embeddingsToBinaryBatch(self, embeddings):
        embeddings = np.array(embeddings)
        
        self.lsh.fit(embeddings)
        
        projected_embeddings = self.lsh.transform(embeddings)
        
        binary_embeddings = (projected_embeddings > 0).astype(int).tolist()
        
        return binary_embeddings
