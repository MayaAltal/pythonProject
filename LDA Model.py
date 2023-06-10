
import glob
import gensim
from gensim.matutils import cossim
from gensim import models
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import pickle

from samilarity import docs
from IPython.display import HTML

from samilarity import vectorizer, calculate_similarity, tfidf_matrix,docs
# Load inverted index
with open('inverted_index_new.pkl', 'rb') as file:
    inverted_index = pickle.load(file)

# تحديد مسار المجلد الذي ترغب في قراءة الملفات منه
folder_path = r"C:\Users\User\Desktop\preprocessed\\preprocessed"

# استخدام glob للعثور على جميع ملفات المجلد
file_paths = glob.glob(folder_path + "/*")

# قراءة المحتوى من كل ملف وتخزينه في قائمة
# قراءة المحتوى من كل ملف وتخزينه في قائمة
contents = []
for file_path in file_paths:
    with open(file_path, 'r') as file:
        content = file.read()
        # قم بمعالجة المحتوى كما تحتاج
        # يمكنك تنفيذ العمليات اللازمة هنا بناءً على المحتوى الفردي لكل ملف

    contents.append([content])  # قم بتخزين المحتوى الفردي لكل ملف في قائمة منفصلة


# Create a dictionary of all the words in the docs
dictionary = gensim.corpora.Dictionary(docs)

# القيام بالإجراءات الإضافية اللازمة على المحتوى المخزن...
# Representing the corpus as a bag of words
corpus = [dictionary.doc2bow(doc) for doc in docs]
# print(corpus)
num_topics = 3
lda_model = gensim.models.LdaModel(corpus=corpus,
                                   id2word=inverted_index,
                                   num_topics=num_topics,
                                   random_state=42,
                                   update_every=1,
                                   chunksize=100,
                                   passes=10,
                                   alpha='auto',
                                   per_word_topics=True)

# show the topics:
topics = lda_model.show_topics(num_topics=num_topics, num_words=10, formatted=False)
for topic in topics:
    print("Topic #{}:".format(topic[0]))
    for word in topic[1]:
        print("\t", word[0], ": ",word[1])

# Step 3: Visualize the LDA model
vis = gensimvis.prepare(lda_model, corpus, dictionary, R=10)
# Save the visualization as an HTML file
pyLDAvis.save_html(vis, r'C:\Users\User\Desktop\lda_visualization.html')
with open(r'C:\Users\User\Desktop\ff.txt', 'r') as file:
    query = file.read()


# إغلاق الملف
file.close()

# query_list = []  # تعريف قائمة فارغة لتخزين التمثيل الجملوني للاستعلامات
# query_list.append(query)  # إضافة التمثيل الجملوني للاستعلام إلى القائمة
#
# print(query_list)  # طباعة القائمة للتحقق من التمثيل الجملوني للاستعلامات المخزنة فيها
# test_doc = [doc.split() for doc in query_list]
# print(test_doc)
# dictionary = gensim.corpora.Dictionary(test_doc)
# print(dictionary)
# test_corpus = [dictionary.doc2bow(doc) for doc in test_doc]
# print(test_corpus)
# doc1 = lda_model.get_document_topics([query], minimum_probability=0)
# doc2 = lda_model.get_document_topics(corpus, minimum_probability=0)
# print("similarity:" ,cossim(doc1, doc2))
query_vector = vectorizer.transform([query]).toarray().flatten()
similarity_scores = calculate_similarity(query_vector, tfidf_matrix.toarray())
print(similarity_scores)
results = []
for i, score in sorted(zip(contents.keys(), similarity_scores[0]), key=lambda x: x[1], reverse=True):
    doc_content = contents[i]
    result = {

        'score': score,
        'content': doc_content
    }
    results.append(result)