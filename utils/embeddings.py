from sklearn.feature_extraction.text import HashingVectorizer

vectorizer = HashingVectorizer(
    n_features=384,
    alternate_sign=False
)

def create_embedding(text):

    vector = vectorizer.transform([text])

    return vector.toarray()[0]