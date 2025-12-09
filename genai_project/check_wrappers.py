import ragas.embeddings


print("Attributes of ragas.embeddings:")
print(dir(ragas.embeddings))

try:
    from ragas.embeddings import LangchainEmbeddingsWrapper
    print("LangchainEmbeddingsWrapper found.")
except ImportError:
    print("LangchainEmbeddingsWrapper NOT found.")
