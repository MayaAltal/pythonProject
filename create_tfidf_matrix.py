from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle

# List of documents
docs = {}

def read_docs_from_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            doc_text = open(file_path, 'r').read()
            doc_id = os.path.splitext(file_name)[0]  # Use the file name without extension as the document ID
            docs[doc_id] = doc_text

def calculate_tfidf_matrix(docs):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs.values())
    return tfidf_matrix, vectorizer

def save_tfidf_matrix(tfidf_matrix, vectorizer, file_path):
    with open(file_path, 'wb') as tfidf_file:
        pickle.dump((tfidf_matrix, vectorizer), tfidf_file)

def build_inverted_index(tfidf_matrix, vectorizer, docs):
    inverted_index = {}

    for doc_id, doc_text in docs.items():
        doc_idx = int(doc_id.split('_')[1])
        feature_values = tfidf_matrix[doc_idx].toarray().flatten()
        for feature_idx, tfidf_value in enumerate(feature_values):
            if tfidf_value > 0:
                feature_name = vectorizer.get_feature_names_out()[feature_idx]
                if feature_name in inverted_index:
                    inverted_index[feature_name][doc_id] = tfidf_value
                else:
                    inverted_index[feature_name] = {doc_id: tfidf_value}

    return inverted_index

# Load documents from the folder
with open(r'C:\Users\User\Desktop\copy.txt', 'r') as file:
    number = file.read()


if number == '1':
    folder_path = r"C:\Users\User\Desktop\preprocessed\preprocessed"
elif number == '2':
    folder_path = r"C:\Users\User\Desktop\docs"

# Read documents from the folder
read_docs_from_folder(folder_path)

# Calculate TF-IDF matrix
tfidf_matrix, vectorizer = calculate_tfidf_matrix(docs)
# print(tfidf_matrix)

# Build inverted index
inverted_index = build_inverted_index(tfidf_matrix, vectorizer, docs)
print(inverted_index)

# Save the TF-IDF matrix to a file
tfidf_file_path = r"C:\Users\User\Desktop\tfidfffffff_matrix_new.pkl"
save_tfidf_matrix(tfidf_matrix, vectorizer, tfidf_file_path)

# Save the inverted index to a file
inverted_index_file_path = 'inverted_index_new.pkl'
with open(inverted_index_file_path, 'wb') as file:
    pickle.dump(inverted_index, file)

# Print the updated docs and inverted index


# Save the docs variable as a pickle file
docs_file_path = 'docs12.pkl'
with open(docs_file_path, 'wb') as file:
    pickle.dump(docs, file)
    print(docs)



