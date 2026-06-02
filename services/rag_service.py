import faiss
import pickle
import numpy as np

from utils.embeddings import create_embedding

_index = None
_documents = None


def _load_index():
    global _index, _documents

    if _index is None:

        _index = faiss.read_index(
            "vector_store/faiss.index"
        )

        with open(
            "vector_store/documents.pkl",
            "rb"
        ) as file:

            _documents = pickle.load(file)

    return _index, _documents


def retrieve_documents(query, top_k=3):

    index, documents = _load_index()

    query_vector = np.array(
        [create_embedding(query)]
    ).astype("float32")

    distances, indices = index.search(
        query_vector,
        top_k
    )

    print("\n====================")
    print("QUESTION:", query)
    print("DISTANCES:", distances)
    print("====================\n")

    results = []

    for idx in indices[0]:

        if 0 <= idx < len(documents):

            chunk = documents[idx]

            print("RETRIEVED CHUNK:")
            print(chunk[:300])
            print("-" * 50)

            results.append(chunk)

    return results