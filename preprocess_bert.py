from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
from unidecode import unidecode
from common import *


def preprocess_bert(publications_en):
    clean_re = re.compile(r'</?[a-z]+(:.+)?>|<([\w-]*|)>')
    http_re = re.compile(r'https?\S+')
    reject = '\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f'

    def preprocess(sentence):
        sentence = str(sentence)
        sentence = unidecode(sentence).replace(reject, '')
        # sentence = sentence.lower()
        # sentence = sentence.replace('{html}', "")
        cleantext = re.sub(clean_re, '', sentence)
        rem_url = re.sub(http_re, '', cleantext)
        # TODO: Remove numbers in word form like 'three', 'fourth'
        # rem_num = re.sub(num_re, '', rem_url)
        # tokens = tokenizer.tokenize(rem_num)
        # filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
        # stem_words = [stemmer.stem(w) for w in filtered_words]
        # lemma_words = [lemmatizer.lemmatize(w) for w in stem_words]
        # return " ".join(lemma_words)
        return rem_url

    publications_en = publications_en.copy()
    publications_en['abstract_text_clean'] = publications_en['abstract_text'].progress_apply(preprocess)
    publications_en = publications_en[publications_en['abstract_text_clean'] != '']
    publications_en = publications_en.drop_duplicates(subset='abstract_text_clean', keep='first')
    save_dataframe(publications_en, "publications_bert")
    return publications_en
