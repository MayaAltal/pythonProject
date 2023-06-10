import glob

import gensim
from gensim.matutils import cossim
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import pickle

from create_tfidf_matrix import vectorizer
from evaluation import read_qrels, read_queries, save_relevant_num, relevant_num_file_path, retrieve_similar_documents
from samilarity import calculate_similarity
# Read relevance judgments (qrels)
qrels_file_paths = glob.glob(r"C:\Users\User\Desktop\qrel.txt")
qrels = read_qrels(qrels_file_paths)
# Read queries from file
queries_file_path = r"C:\Users\User\Desktop\queries.txt"
queries = read_queries(queries_file_path)
# قراءة النصوص من ملف pkl
with open('docs12.pkl', 'rb') as file:
    docs = pickle.load(file)

# قراءة inverted index من ملف pkl
with open('inverted_index_new.pkl', 'rb') as file:
    inverted_index = pickle.load(file)

# تحويل inverted index إلى الهيكل البيانات المطلوب
inverted_index_dict = {}
for word, freq in inverted_index.items():
    inverted_index_dict[word] = freq

# استخدام inverted index كـ dictionary مع gensim
dictionary = gensim.corpora.Dictionary([inverted_index_dict.keys()])

# تحويل المستندات إلى الهيكل البيانات المطلوب
docs_dict = {}
for doc_id, doc_content in docs.items():
    docs_dict[doc_id] = doc_content

# Representing the corpus as a bag of words
all_words = []
for doc_content in docs_dict.values():
    all_words.extend(doc_content.split())

dictionary = gensim.corpora.Dictionary([all_words])
corpus = [dictionary.doc2bow(doc_content.split()) for doc_content in docs_dict.values()]

num_topics = 5
passes = 20

# تحسين النموذج LDA
lda_model = gensim.models.LdaModel(corpus=corpus,
                                   id2word=dictionary,
                                   num_topics=num_topics,
                                   random_state=42,
                                   update_every=1,
                                   chunksize=100,
                                   passes=passes,
                                   alpha='auto',
                                   per_word_topics=True)

# عرض المواضيع
topics = lda_model.show_topics(num_topics=num_topics, num_words=10, formatted=False)
for topic in topics:
    print("Topic #{}:".format(topic[0]))
    for word in topic[1]:
        print("\t", word[0], ": ", word[1])

# حفظ التصور البصري للنموذج
vis = gensimvis.prepare(lda_model, corpus, dictionary, R=30)
pyLDAvis.save_html(vis, 'lda_visualization.html')

# قراءة الاستعلام من ملف
# with open(r'C:\Users\User\Desktop\ff.txt', 'r') as file:
#     query = file.read()
for query_id, query in queries.items():
    print("Query ID:", query_id)
    print("Query:", query)

    if query_id not in qrels:
        print("No relevance judgments found for this query.")
        continue
    # تحويل الاستعلام إلى تمثيله الجملوني
    query_vector = dictionary.doc2bow(query.split())
    print("ااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااا")

    # حساب التشابهية بين الاستعلام والمستندات
    similarity_scores = []
    for doc_vector in corpus:
        similarity_score = cossim(query_vector, doc_vector)
        similarity_scores.append(similarity_score)

    # إنشاء قائمة لتخزين النتائج
    results = []
    for i, score in sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True):
        doc_index = list(docs_dict.keys())[i]
        doc_content = docs_dict[doc_index]
        result = {
            'doc_index': doc_index,
            'score': score,
            'content': doc_content
        }
        results.append(result)

        if doc_index in qrels.get(query_id, {}):

            print("......................................")

    # طباعة النتائج
    for result in results:
        print("Document Index:", result['doc_index'])
        print("Score:", result['score'])
        print("Content:", result['content'])