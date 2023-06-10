import pandas as pd
import os
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import wordnet
import datefinder
import re
from typing import Any
from datetime import datetime
from dateutil.parser import parse
from nltk.corpus import wordnet
import glob


abbreviations = {
    'Dr.': 'Doctor',
    'Mr.': 'Mister',
    'Mrs.': 'Misess',
    'Ms.': 'Misess',
    'Jr.': 'Junior',
    'Sr.': 'Senior',
    'U.S': 'UNITED STATES',
    'U-S': 'UNITED STATES',
    'U_K': 'UNITED KINGDOM',
    'U_S': 'UNITED STATES',
    'U.K': 'UNITED KINGDOM',
    'U.S': 'UNITED STATES',
    'VIETNAM': 'VIET NAM',
    'VIET NAM': 'VIET NAM',
    'U-N': 'NITED NATIONS',
    'U_N': 'NITED NATIONS',
    'U.N': 'NITED NATIONS',
    'UK': 'UNITED KINGDOM',
    'US': 'UNITED STATES',
    'U-K': 'UNITED KINGDOM',
    'mar': 'March',
    'march': 'March',
    'jan': 'January',
    'anuary': 'January',
    'feb': 'February',
    'february': 'February',
    'apr': 'April',
    'april': 'April',
    'jun': 'June',
    'june': 'June',
    'jul': 'July',
    'july': 'July',
    'dec': 'December',
    'december': 'December',
    'nov': 'November',
    'november': 'November',
    'oct': 'October',
    'october': 'October',
    'sep': 'September',
    'september': 'September',
    'aug': 'August',
    'august': 'August',
}

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def stem_words(txt):
    stems = [stemmer.stem(word) for word in txt]
    return stems


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def handle(text):
    REGEX_1 = r"(([12]\d|30|31|0?[1-9])[/-](0?[1-9]|1[0-2])[/.-](\d{4}|\d{2}))"
    REGEX_2 = r"((0?[1-9]|1[0-2])[/-]([12]\d|30|31|0?[1-9])[/.-](\d{4}|\d{2}))"
    REGEX_3 = r"((\d{4}|\d{2})[/-](0?[1-9]|1[0-2])[/-]([12]\d|30|31|0?[1-9]))"
    REGEX_4 = r"((January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec) ([12]\d|30|31|0?[1-9]),? (\d{4}|\d{2}))"
    REGEX_5 = r"(([12]\d|30|31|0?[1-9]) (January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec),? (\d{4}|\d{2}))"
    REGEX_6 = r"((\d{4}|\d{2}) ,?(January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec) ([12]\d|30|31|0?[1-9]))"
    REGEX_7 = r"((January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec),? (\d{4}|\d{2}))"

    COMBINATION_REGEX = "(" + REGEX_1 + "|" + REGEX_2 + "|" + REGEX_3 + "|" + \
                       REGEX_4 + "|" + REGEX_5 + "|" + REGEX_6 + ")"

    for key, value in abbreviations.items():
        text = re.sub(r'\b{}\b'.format(re.escape(key)), value, text, flags=re.IGNORECASE)

    all_dates = re.findall(COMBINATION_REGEX, text)

    for s in all_dates:
        try:
            date = datetime.strptime(s[0], "%d %B %Y")
        except ValueError:
            continue  # Skip invalid dates

        new_date = date.strftime("%d-%m-%Y")
        text = text.replace(s[0], new_date)


    text = re.sub(r'[^-\w\s]', '', text)

    return text


folder_path = "C:/Users/User/Desktop/new_directory"

file_paths = glob.glob(folder_path + "/*.txt")
for file_path in file_paths:
    with open(file_path, 'r') as file:
        text = file.read()

        print("-------------------")
        unified_text = handle(text)
        print("File content after unification:")
        # print(unified_text)

        sentences = sent_tokenize(unified_text)
        tokens = word_tokenize(unified_text)
        print("-------------------")
        print("File:", file_path)
        print("TOKENIZE:", tokens)

        tokens = [w.lower() for w in tokens]
        print("Lowercase:", tokens)

        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
        print("Remove stopwords:", filtered_tokens)

        stem_word = stem_words(filtered_tokens)
        print("stemming:", stem_word)

        listToStr = ' '.join([str(elem) for elem in stem_word])
        pos_tagged = nltk.pos_tag(nltk.word_tokenize(listToStr))

        print("pos_tagged:", pos_tagged)

        wordnet_tagged = [(word, pos_tagger(tag)) for word, tag in pos_tagged]
        print("wordnet_tagged:", wordnet_tagged)

        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                lemmatized_sentence.append(word)
            else:
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        lemmatized_sentence = " ".join(lemmatized_sentence)

        print("lemmatized_sentence:", lemmatized_sentence)

        output_file_path = file_path.replace("C:/Users/User/Desktop/new_directory", "C:/Users/User/Desktop/UUU")
        with open(output_file_path, "w") as output_file:
            output_file.write(lemmatized_sentence)

        print("Result has been stored in:", output_file_path)
