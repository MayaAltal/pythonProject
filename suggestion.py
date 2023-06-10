import glob
from fast_autocomplete import AutoComplete
import ipywidgets as widgets
from IPython.display import display
import nltk
from spellchecker import SpellChecker
from nltk.corpus import wordnet

# Initialize the spellchecker
spellchecker = nltk.corpus.words.words()


def read_file(file_path):
    queries = []
    with open(file_path, 'r') as file:
        file_queries = file.readlines()
        for line in file_queries:
            query = line.split("\t")[1]  # يتم الحصول على الجزء الثاني من السطر (نص الاستعلام)
            queries.append(query.strip())  # يتم إضافة الاستعلام إلى القائمة

    return queries


# تحديث المتغير folder_path ليكون مسار الملف المطلوب
file_path = "C:/Users/User/Desktop/queries.txt"
queries = read_file(file_path)


def complete(query, folder_path):
    queries = read_file(folder_path)
    words = {}
    for value in queries:
        value = value.strip()  # Remove leading/trailing whitespace
        new_key_values_dict = {value: {}}
        words.update(new_key_values_dict)

    autocomplete = AutoComplete(words=words)
    suggestions = autocomplete.search(query, max_cost=10, size=10)

    return suggestions


def suggest_non_start_words(query, folder_path):
    queries = read_file(folder_path)
    suggestions = []
    for value in queries:
        value = value.strip()  # Remove leading/trailing whitespace
        if query in value and not value.startswith(query):
            suggestions.append(value)

    return suggestions


def suggest_spelling_corrections(query):
    tokens = nltk.word_tokenize(query)
    spellchecker = SpellChecker()
    corrections = []
    for token in tokens:
        if token.lower() not in spellchecker:
            correction = spellchecker.correction(token)
            corrections.append(correction)

    return corrections


# استخدام الدالة expand_query لتوسيع استعلام المستخدم الأصلي. تقوم الدالة
# بتحليل استعلام المستخدم والبحث عن المرادفات
#  والمتضادات والكلمات الأكثر تحديدًا للمفهوم
#  وإضافتها إلى استعلام المستخدم الأصلي.
def expand_query(query):
    synonyms = set()
    antonyms = set()
    hypernyms = set()

    # Find synonyms المرادفات
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())

    # Find antonyms المضادات
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms.add(lemma.antonyms()[0].name())

    # Find hypernyms
    for word in query.split():
        for syn in wordnet.synsets(word):
            for hyper in syn.hypernyms():
                for lemma in hyper.lemmas():
                    hypernyms.add(lemma.name())

    expanded_query = query.split() + list(synonyms) + list(antonyms) + list(hypernyms)

    return expanded_query


# folder_path = "C:/Users/User 2004/Desktop/New folder (2)"

def on_text_changed(query):
    print("Autocomplete suggestions:")
    autocomplete_suggestions = complete(query, file_path)
    for suggestion in autocomplete_suggestions:
        print(suggestion)

    print("\nNon-start words suggestions:")
    non_start_words_suggestions = suggest_non_start_words(query, file_path)
    for suggestion1 in non_start_words_suggestions:
        print(suggestion1)

    print("\nSpelling corrections:")
    spelling_corrections = suggest_spelling_corrections(query)
    for correction in spelling_corrections:
        print(correction)

    print("\nexpanded_query :")
    expanded_query = expand_query(query)
    for expanded in expanded_query:  # استخدم expanded_query بدلاً من expand_query
        print(expanded)

    all_suggestions = autocomplete_suggestions + non_start_words_suggestions + spelling_corrections + expanded_query
    print(all_suggestions)  # طباعة الاقتراحات
    return all_suggestions

#
#
# on_text_changed({'new': 'how do hwllo'})


