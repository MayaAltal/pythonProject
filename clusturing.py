import glob

import numpy as np

from samilarity import calculate_similarity, vectorizer, tfidf_matrix, docs
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from create_tfidf_matrix import calculate_tfidf_matrix
from evaluation import read_qrels, read_queries, save_relevant_num, relevant_num_file_path, retrieve_similar_documents

# Read relevance judgments (qrels)
qrels_file_paths = glob.glob(r"C:\Users\User\Desktop\qrel.txt")
qrels = read_qrels(qrels_file_paths)
# Read queries from file
queries_file_path = r"C:\Users\User\Desktop\queries.txt"
queries = read_queries(queries_file_path)

# تحميل المستندات من ملف docs1111.pkl
with open('docs12.pkl', 'rb') as file:
    docs = pickle.load(file)

# تحويل المستندات إلى صورة مرجعية باستخدام TfidfVectorizer
# Calculate TF-IDF matrix
tfidf_matrix, vectorizer = calculate_tfidf_matrix(docs)

# # الاستعلام
# with open(r'C:\Users\User\Desktop\ff.txt', 'r') as file:
#     query = file.read()

# عدد الفئات المطلوبة
num_clusters = 4
#
# # إنشاء نموذج التجميع باستخدام K-Means
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(tfidf_matrix)
#
# # تحويل الاستعلام إلى نموذج tf-idf
# query_vector = vectorizer.transform([query]).toarray().flatten()
# # print(query_vector)
#
# similarity_scores = calculate_similarity(query_vector, tfidf_matrix.toarray())

# results = []
# for doc_id, score in sorted(zip(docs.keys(), similarity_scores[0]), key=lambda x: x[1], reverse=True):
#     doc_content = docs[doc_id]
#     result = {
#         'doc_id': doc_id,
#         'score': score,
#         'content': doc_content
#     }
#     results.append(result)
# #
# for result in results:
#     print("Document Index:", result['doc_id'])
#     print("Score:", result['score'])
#     print("Content:", result['content'])
#     print("==============================")
for i in range(kmeans.n_clusters):
    cluster_indices = np.where(kmeans.labels_ == i)[0]
    cluster_docs = [list(docs.keys())[doc_index] for doc_index in cluster_indices]
    cluster_texts = [list(docs.values())[doc_index] for doc_index in cluster_indices]

    print("Cluster #{} documents:".format(i))
    for doc_id, doc_text in zip(cluster_docs, cluster_texts):
        print("- Document ID:", doc_id)
        print("  Content:", doc_text)
    print("==============================")

# عرض المستندات في كل مجموعة
# for i in range(kmeans.n_clusters):
#     cluster_docs = [doc for j, doc in enumerate(docs) if kmeans.labels_[j] == i]
#     print("Cluster #{} documents:".format(i))
#     for doc in cluster_docs:
#         print("- ", doc)
#     print("==============================")

# plot the data points with different colors based on their assigned cluster
# plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('K-Means Clustering')
#
# # Show the plot
# plt.show()
for query_id, query in queries.items():
        print("Query ID:", query_id)
        print("Query:", query)

        if query_id not in qrels:
            print("No relevance judgments found for this query.")
            continue

        query_vector = vectorizer.transform([query]).toarray().flatten()

        # Calculate similarity using inverted index
        similarity_scores = calculate_similarity(query_vector, tfidf_matrix.toarray())

        # Retrieve similar documents
        num_relevant, num_returned, precision, Total, relevant_documents = retrieve_similar_documents(
            query_id, query, qrels, similarity_scores, tfidf_matrix, docs, vectorizer)

        # Save the results
        save_relevant_num(relevant_num_file_path, query_id, query, num_relevant, num_returned, precision, Total,relevant_documents)