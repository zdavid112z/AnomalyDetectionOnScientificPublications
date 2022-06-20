from spacy_langdetect import LanguageDetector
from spacy.language import Language
import spacy
from common import *


def init_nlp():
    @Language.factory(
        'language_detector2',
    )
    def create_language_detector(nlp, name):
        return LanguageDetector()

    nlp = spacy.load('en_core_web_trf')
    # nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe('language_detector2', name='language_detector', last=True, validate=False)
    return nlp


def load_lang_detects(publications: pd.DataFrame, nlp):
    def detect_language(text):
        if text is not None and type(text) == str and len(text) > 0:
            return pd.Series(nlp(text)._.language)
        return pd.Series({'language': '??', 'score': 0})

    print("Detecting languages...")
    lang_detects = publications['abstract_text'].progress_apply(detect_language)

    return lang_detects


# removes publications with non-english or missing abstracts
def preprocess_publications_common(publications: pd.DataFrame, nlp):
    lang_detects = load_lang_detects(publications, nlp)
    publications['language'] = lang_detects['language']
    publications['lang_score'] = lang_detects['score']
    publications = publications[publications['language'] == 'en'][publications['lang_score'] > 0.99]
    save_dataframe(publications, "publications_en")
    return publications
