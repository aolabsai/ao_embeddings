import embedding_bucketing.embedding_model_test as em
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
import ao_core as ao
import ao_arch as ar
from config import openai_api_key


description = "Basic Semantics"
arch_i = [64]     # 3 neurons, 1 in each of 3 channels, corresponding to Food, Chemical-A, Chemical-B (present=1/not=0)
arch_z = [10]           # corresponding to Open=1/Close=0
arch_c = []           # adding 1 control neuron which we'll define with the instinct control function below
connector_function = "full_conn"

# To maintain compatibility with our API, do not change the variable name "Arch" or the constructor class "ar.Arch" in the line below
Arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, description)


# Initialize Agent
Agent = ao.Agent(Arch, save_meta=True)

# Configure embedding model
em.config(openai_api_key)

# List of words
# words = ["dog", "dog food", "spaceship", "space"]
words = ["dog", "cat", "house", "brick"]

embeddings = []

# Get embeddings for all words
for word in words:
    embedding = em.get_embedding(word) 
    embeddings.append(embedding)

embeddings = np.array(embeddings)

    # Initialize and fit LSH model
lsh = GaussianRandomProjection(n_components=64)
lsh.fit(embeddings)  # Fit once

binary_embeddings = []
for embedding in embeddings:
    projected_embedding = lsh.transform(embedding.reshape(1, -1)) 
    binary_embedding = (projected_embedding > 0).astype(int).flatten().tolist()  # Convert to list
    binary_embeddings.append(binary_embedding)


print("Binary Embeddings: ", binary_embeddings)
print("Length encoding: ", len(binary_embeddings[0]))

print("0 ", Agent.next_state(binary_embeddings[0], LABEL=np.ones(10), DD=False, print_result=True, unsequenced=True))
Agent.reset_state()
print("2", Agent.next_state(binary_embeddings[2], LABEL=np.zeros(10), DD=False, print_result=True, unsequenced=True))

# we should add reset states, and if we add reset states .next_state should be set with unsequenced=True
Agent.reset_state()
print("1 ", Agent.next_state(binary_embeddings[1], DD=False, print_result=True, unsequenced=True))
Agent.reset_state()
print("3 ", Agent.next_state(binary_embeddings[2], DD=False, print_result=True, unsequenced=True))



