import faiss
import pickle
import numpy as np

from utils.embeddings import create_embedding

_index = None
_documents = None


def load_rag():

    global _index
    global _documents

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


def retrieve_documents(
    query,
    top_k=3
):

    index, documents = load_rag()

    query_embedding = np.array(
        [create_embedding(query)]
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for idx in indices[0]:

        if (
            idx >= 0 and
            idx < len(documents)
        ):

            results.append(
                documents[idx]
            )

    return results