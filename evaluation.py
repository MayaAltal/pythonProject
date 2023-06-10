import numpy as np
import glob
from samilarity import calculate_similarity, vectorizer, tfidf_matrix, docs
from processquery import process
def read_qrels(qrels_file_paths):
    qrels = {}
    for qrels_file_path in qrels_file_paths:
        with open(qrels_file_path, 'r') as qrels_file:
            for line in qrels_file:
                line_parts = line.strip().split(' ')
                if len(line_parts) != 4:
                    print("Invalid line format:", line)
                    continue

                query_id, _, doc_id, relevance = line_parts
                query_id = int(query_id)
                doc_idd = doc_id.split('_')[0]  # Extract the document ID without the suffix


                if query_id in qrels:
                    qrels[query_id][doc_idd] = relevance
                else:
                    qrels[query_id] = {doc_idd: relevance}

    return qrels


def read_queries(queries_file_path):
    queries = {}
    with open(queries_file_path, 'r') as queries_file:
        for line in queries_file:

            line_parts = line.strip().split(None, 1)
            if len(line_parts) < 2:
                continue  # Skip lines that don't have the expected format
            query_id = int(line_parts[0])
            query = line_parts[1]
            query=process(query)
            queries[query_id] = query

    return queries


def retrieve_similar_documents(query_id, query, qrels, similarity_scores, tfidf_matrix, docs, vectorizer):
    sorted_indices = np.argsort(similarity_scores, axis=1)[0, ::-1]
    threshold = 0.00000001
    num_relevant = 0
    num_returned = 0
    relevant_documents = []

    for idx in sorted_indices:
        similarity_score = similarity_scores[0, idx]
        doc_id = list(docs.keys())[idx]  # Retrieve the document ID using the index
        doc_idd = doc_id.split('_')[0]  # Extract the document ID without the suffix

        if similarity_score > 0.1:
            if similarity_score >= threshold and doc_idd in qrels.get(query_id, {}):
                num_relevant += 1
                relevant_documents.append(
                    f"Similarity Score: {similarity_score}, Document ID: {doc_id}, Document: {docs[doc_id]}")

            num_returned += 1

    precision = num_relevant / num_returned if num_returned > 0 else 0.0
    print("Precision:", num_relevant)

    Total = len(sorted_indices)

    return num_relevant, num_returned, precision, Total, relevant_documents


def save_relevant_num(relevant_num_file_path, query_id, query, num_relevant, num_returned, precision, Total,
                      relevant_documents):
    with open(relevant_num_file_path, 'a+') as relevant_num_file:
        relevant_num_file.write(f"Query ID: {query_id}\n")
        relevant_num_file.write(f"Query: {query}\n")
        relevant_num_file.write(f"Relevant Documents: {num_relevant}\n")
        relevant_num_file.write(f"Total Returned Documents: {num_returned}\n")
        relevant_num_file.write(f"Precision: {precision}\n")
        relevant_num_file.write(f"Total: {Total}\n")
        relevant_num_file.write("\n".join(relevant_documents))
        relevant_num_file.write("\n\n")






# Read relevance judgments (qrels)
qrels_file_paths = glob.glob(r"C:\Users\User\Desktop\qrel.txt")
qrels = read_qrels(qrels_file_paths)
# Read queries from file
queries_file_path = r"C:\Users\User\Desktop\queries.txt"
queries = read_queries(queries_file_path)

# Create relevant_num_file to store the results
relevant_num_file_path = r"C:\Users\User\Desktop\relevanttt_num_queries.txt"
with open(relevant_num_file_path, 'w') as relevant_num_file:
    for query_id, query in queries.items():
        print("Query ID:", query_id)
        print("Query:", query)

        if query_id not in qrels:
            print("No relevance judgments found for this query.")
            continue

        # Tokenize and transform the query
        query_vector = vectorizer.transform([query]).toarray().flatten()

        # Calculate similarity using inverted index
        similarity_scores = calculate_similarity(query_vector, tfidf_matrix.toarray())

        # Retrieve similar documents
        num_relevant, num_returned, precision, Total, relevant_documents = retrieve_similar_documents(
            query_id, query, qrels, similarity_scores, tfidf_matrix, docs, vectorizer)

        # Save the results
        save_relevant_num(relevant_num_file_path, query_id, query, num_relevant, num_returned, precision, Total,
                          relevant_documents)
