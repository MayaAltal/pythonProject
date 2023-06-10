from flask import Flask, request, jsonify
from processquery import process
from samilarity import calculate_similarity, vectorizer, tfidf_matrix, docs
from suggestion import on_text_changed


app = Flask(__name__)

with open(r'C:\Users\User\Desktop\ff.txt', 'r') as file:
    query = file.read()

@app.route('/query', methods=['POST'])
# Tokenize and transform the query

def process_query():
    query = request.json['query']  # Assuming the query is sent as a JSON object with the key 'query'
    number = request.json['number']
    query=process(query)


    # إغلاق الملف

    output_file = r'C:\Users\User\Desktop\ff.txt'
    with open(output_file, 'w') as file:
        file.write(query)
        file.close()
        output_file = r'C:\Users\User\Desktop\copy.txt'
        with open(output_file, 'w') as file:
            file.write(number)
            file.close()

    query_vector = vectorizer.transform([query]).toarray().flatten()
    similarity_scores = calculate_similarity(query_vector, tfidf_matrix.toarray())

    # Sort the results by similarity score
    results = []
    for doc_id, score in sorted(zip(docs.keys(), similarity_scores[0]), key=lambda x: x[1], reverse=True):
        doc_content = docs[doc_id]
        result = {
            'doc_id': doc_id,
            'score': score,
            'content': doc_content
        }
        results.append(result)

    return jsonify(results)

@app.route('/suggestion', methods=['POST'])
def suggestion():
    query = request.json['query']
    result = on_text_changed(query)
    return jsonify(result)


if __name__ == '__main__':
    app.run(port=8000)
