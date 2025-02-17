import ao_core as ao
import ao_arch as ar
from config import openai_api_key
import binaryEmbeddings as be
import numpy as np

description = "Basic Semantics"
arch_i = [64]     # 3 neurons, 1 in each of 3 channels, corresponding to Food, Chemical-A, Chemical-B (present=1/not=0)
arch_z = [10]           # corresponding to Open=1/Close=0
arch_c = []           # adding 1 control neuron which we'll define with the instinct control function below
connector_function = "full_conn"

# To maintain compatibility with our API, do not change the variable name "Arch" or the constructor class "ar.Arch" in the line below
Arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, description)


# Initialize Agent
Agent = ao.Agent(Arch, save_meta=True)

embeddingToBinary = be.binaryEmbeddings(openai_api_key)

# List of words
words = ["dog", "pets", "house", "brick"]
#words = ["Hat", "Scarf", "house", "brick"]

binary_embeddings = []

# Get embeddings for all words
for word in words:
    embedding = embeddingToBinary.get_embedding(word)
    binary_embedding = embeddingToBinary.embeddingToBinary(embedding)
    binary_embeddings.append(binary_embedding)

print("Binary Embeddings: ", binary_embeddings)
print("Length encoding: ", len(binary_embeddings[0]))

Agent.next_state(binary_embeddings[0], LABEL=np.ones(10), DD=False, unsequenced=True)
Agent.reset_state()
Agent.next_state(binary_embeddings[2], LABEL=np.zeros(10), DD=False, unsequenced=True)

# we should add reset states, and if we add reset states .next_state should be set with unsequenced=True
Agent.reset_state()
print("---1---")
Agent.next_state(binary_embeddings[1], DD=False, print_result=True, unsequenced=True)
Agent.reset_state()
print("---3---")
Agent.next_state(binary_embeddings[3], DD=False, print_result=True, unsequenced=True)



