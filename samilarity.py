import pickle

from sklearn.metrics.pairwise import cosine_similarity

with open(r"C:\Users\User\Desktop\tfidfffffff_matrix_new.pkl", 'rb') as file:
    tfidf_matrix, vectorizer = pickle.load(file)

# Load inverted index
with open('inverted_index_new.pkl', 'rb') as file:
    inverted_index = pickle.load(file)

# Load docs
with open('docs12.pkl', 'rb') as file:
    docs = pickle.load(file)


def calculate_similarity(query_vector, tfidf_matrix):
    similarity_scores = cosine_similarity(query_vector.reshape(1, -1), tfidf_matrix)
    return similarity_scores
print(tfidf_matrix)