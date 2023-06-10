import glob
import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from create_tfidf_matrix import calculate_tfidf_matrix
from evaluation import read_qrels, read_queries, save_relevant_num, relevant_num_file_path


def retrieve_similar_documents(query_id, query, qrels, similarity_scores, tfidf_matrix, docs, vectorizer):
    sorted_indices = np.argsort(similarity_scores.reshape(1, -1), axis=1)[0, ::-1]
    threshold = 0
    num_relevant = 0
    returned_docs = []

    for i, idx in enumerate(sorted_indices):
        doc_id = list(docs.keys())[idx]
        if doc_id in qrels[query_id]:
            num_relevant += 1
        precision = num_relevant / (i + 1)
        returned_docs.append(doc_id)
        if precision >= threshold and i + 1 >= 5:
            break

    num_returned = i + 1
    relevant_documents = [doc_id for doc_id in returned_docs if doc_id in qrels[query_id]]
    Total = len(qrels[query_id])

    return num_relevant, num_returned, precision, Total, relevant_documents

# Read relevance judgments (qrels)
qrels_file_paths = glob.glob(r"C:\Users\Riham\Desktop\.ir project\qrel.txt")
qrels = read_qrels(qrels_file_paths)
# Read queries from file
queries_file_path = r"C:\Users\Riham\Desktop\.ir project\queries.txt"
queries = read_queries(queries_file_path)
with open(r'C:\Users\Riham\Desktop\.ir project\docs12.pkl', 'rb') as file:
    docs = pickle.load(file)
# تحويل المستندات إلى صورة مرجعية باستخدام TfidfVectorizer
tfidf_matrix, vectorizer = calculate_tfidf_matrix(docs)
# عدد الفئات المطلوبة
num_clusters = 2
# # إنشاء نموذج التجميع باستخدام K-Means
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(tfidf_matrix)

for query_id, query in queries.items():
    print("Query ID:", query_id)
    print("Query:", query)

    if query_id not in qrels:
        print('yuuu******************************************')
        print("No relevance judgments found for this query.")
        continue

    query_vector = vectorizer.transform([query]).toarray().flatten()

    # Calculate similarity between query and each cluster
    for i, cluster_center in enumerate(kmeans.cluster_centers_):
        # Calculate similarity between query and cluster center using cosine_similarity
        similarity_score = cosine_similarity(query_vector.reshape(1, -1), cluster_center.reshape(1, -1))[0, 0]
        # Get documents matrix in the cluster
        cluster_docs = tfidf_matrix[kmeans.labels_ == i]
        # Print cluster ID and similarity score
        print("Cluster ID: {}".format(i))
        print("Similarity Score between Query and Cluster_center-{}: {}".format(i, similarity_score))
        # Get indices of rows that match the condition kmeans.labels_ == i
        doc_indices = np.where(kmeans.labels_ == i)[0]
        # Get document IDs associated with the cluster
        doc_ids = [list(docs.keys())[j] for j in doc_indices]
        # Print document content and similarity scores for each document in the cluster
        for j, (doc_id, doc_content) in enumerate(docs.items()):
            if doc_id in doc_ids:
                # Calculate similarity between document and cluster center using cosine_similarity
                doc_similarity_score = cosine_similarity(tfidf_matrix[j], query_vector.reshape(1, -1))[0, 0]
                # Print document ID, similarity score, and document content
                print("Document ID in Cluster-{}, Document-{}: {}".format(i, j, doc_id))
                print("Similarity Score between Document-{} in Cluster-{} and the query: {}".format(j, i,
                                                                                                    doc_similarity_score))
                print("Document Content in Cluster-{}, Document-{}: {}".format(i, j, doc_content))
                print("\n")

                # Retrieve similar documents
                num_relevant, num_returned, precision, Total, relevant_documents = retrieve_similar_documents(query_id,
                                                                                                              query,
                                                                                                              qrels,
                                                                                                              doc_similarity_score,
                                                                                                              cluster_docs,
                                                                                                              docs,
                                                                                                              vectorizer)

    # Save the results
    save_relevant_num(relevant_num_file_path, query_id, query, num_relevant, num_returned, precision, Total,
                      relevant_documents)


