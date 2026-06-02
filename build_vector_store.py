import os
import faiss
import pickle
import numpy as np

from utils.chunking import chunk_text
from utils.embeddings import create_embedding


documents = []

pubmed_files = [
    "datasets/pubmed/heart_disease.txt",
    "datasets/pubmed/pneumonia.txt"
]

for file_path in pubmed_files:

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

        chunks = chunk_text(
            text,
            chunk_size=500
        )

        documents.extend(chunks)


embeddings = []

for chunk in documents:

    embedding = create_embedding(
        chunk
    )

    embeddings.append(
        embedding
    )

embeddings = np.array(
    embeddings
).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    embeddings
)

os.makedirs(
    "vector_store",
    exist_ok=True
)

faiss.write_index(
    index,
    "vector_store/faiss.index"
)

with open(
    "vector_store/documents.pkl",
    "wb"
) as file:

    pickle.dump(
        documents,
        file
    )

print(
    f"Stored {len(documents)} chunks"
)