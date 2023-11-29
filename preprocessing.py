import re
import unicodedata

import nltk as nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
#from nltk import word_tokenize
#from pyinstrument import Profiler as pyinstrument_profiler

from contractions import CONTRACTION_MAP


tokenizer = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()

SW_SET = nltk.corpus.stopwords.words('english')
SW_SET.remove('no')
SW_SET.remove('not')


# # Cleaning Text - strip HTML

# # Removing accented characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


# # Expanding Contractions
def expand_match(contraction, contraction_mapping=CONTRACTION_MAP):
    match = contraction.group(0)
    first_char = match[0]
    expanded_contraction = contraction_mapping.get(match) or contraction_mapping.get(match.lower())

    expanded_contraction = first_char + expanded_contraction[1:]
    return expanded_contraction

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile(f"({'|'.join(contraction_mapping.keys())})", flags=re.IGNORECASE | re.DOTALL)

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


# # Removing Special Characters
def remove_special_characters(text):
    text = re.sub('[^a-zA-Z0-9\s]', ' ', text)
    return text


# # Lemmatizing text
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

# # Removing Stopwords
def remove_stopwords(text, is_lower_case=True):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in SW_SET]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in SW_SET]
    return ' '.join(filtered_tokens)


def process_text(text, urls_stripping=True, contraction_expansion=True,
                 accented_char_removal=True, text_lower_case=True,
                 text_lemmatization=True, special_char_removal=True, stopword_removal=True):
    # Excluding html tags

    text = re.sub(r'<[^<>]+>', " ", text)
    # Excludind urls
    if urls_stripping:
        text = re.sub('http\S+', ' ', text)

    if accented_char_removal:
        text = remove_accented_chars(text)

    if contraction_expansion:
        text = expand_contractions(text)


    # converting to lower case
    if text_lower_case:
        text = text.lower()
    # remove extra newlines
    text = re.sub(r'[\r|\n|\r\n]+', ' ', text)
    # insert spaces between special characters to isolate them
    #special_char_pattern = re.compile(r'([{.(-)!}])')
    #text = special_char_pattern.sub(" \\1 ", text)

    if special_char_removal:
        text = remove_special_characters(text)

    # remove extra whitespace
    text = re.sub('\s+', ' ', text)

    if stopword_removal:
        text = remove_stopwords(text, is_lower_case=text_lower_case)

    if text_lemmatization:
        text = lemmatize_text(text)

    return text.strip()

def preprocess_dataset(train_filepath, output_filepath, urls_stripping=True,
                       contraction_expansion=True,
                       accented_char_removal=True, text_lower_case=True,
                       text_lemmatization=True, special_char_removal=True, stopword_removal=True):
    # Import training data
    df_movies_train = pd.read_csv(train_filepath)
    df_movies_train.drop(['id'], axis=1, inplace=True)
    #df_movies_train = df_movies_train.head(100)
    # preprocess
    df_movies_train['text'] = df_movies_train['text'].apply(
        lambda txt: process_text(txt, urls_stripping=urls_stripping,
                                 contraction_expansion=contraction_expansion,
                                 accented_char_removal=accented_char_removal, text_lower_case=text_lower_case,
                                 text_lemmatization=text_lemmatization, special_char_removal=special_char_removal,
                                 stopword_removal=stopword_removal))

    df_movies_train.to_csv(output_filepath)
    return output_filepath

def preprocess_dataset(train_filepath, output_filepath="output.csv", urls_stripping=True,
                       contraction_expansion=True,
                       accented_char_removal=True, text_lower_case=True,
                       text_lemmatization=True, special_char_removal=True, stopword_removal=True,to_csv=True):
    # Import training data
    df_movies_train = pd.read_csv(train_filepath)
    if to_csv:
        df_movies_train.drop(['id'], axis=1, inplace=True)
    #df_movies_train = df_movies_train.head(100)
    # preprocess
    df_movies_train['text'] = df_movies_train['text'].apply(
        lambda txt: process_text(txt, urls_stripping=urls_stripping,
                                 contraction_expansion=contraction_expansion,
                                 accented_char_removal=accented_char_removal, text_lower_case=text_lower_case,
                                 text_lemmatization=text_lemmatization, special_char_removal=special_char_removal,
                                 stopword_removal=stopword_removal))

    if not to_csv:
        return df_movies_train
    df_movies_train.to_csv(output_filepath)
    return output_filepath

def main():
    for TEXT_LEMMATIZATION in [0]:
        for STOPWORD_REMOVAL in [0]:
            for CONTRACTION_EXPANSION in [0]:
                namefile =f"data/train_preprocessed__cntrc_exp_{CONTRACTION_EXPANSION}__lmmtiz_{TEXT_LEMMATIZATION}__stpwrd_rm_{STOPWORD_REMOVAL}.csv"
                print(f"start preprocess {namefile}")
                preprocess_dataset("data/train.csv",
                                   f"data/train_preprocessed__cntrc_exp_{CONTRACTION_EXPANSION}__lmmtiz_{TEXT_LEMMATIZATION}__stpwrd_rm_{STOPWORD_REMOVAL}.csv",
                                   text_lemmatization=TEXT_LEMMATIZATION, stopword_removal=STOPWORD_REMOVAL, contraction_expansion=CONTRACTION_EXPANSION)
                print(f"done : {namefile}")


if __name__ == '__main__':
    #profiler = pyinstrument_profiler()

    #profiler.start()
    main()
    #profiler.stop()

    #profiler.print(color=True,show_all=True,timeline=True)
