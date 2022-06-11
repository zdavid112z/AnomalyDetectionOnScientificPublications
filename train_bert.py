import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
from spacy_langdetect import LanguageDetector
from spacy.language import Language
import spacy
from tqdm import tqdm
from tqdm.notebook import tqdm
import pickle
import seaborn as sns
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from unidecode import unidecode
import gensim
import logging
import gensim.corpora as corpora
from pprint import pprint
import common
import user_profile
import wordcloud
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


class BERTConfig:
    def __init__(self):
        self.embedding_model = "all-distilroberta-v1"
        self.device = "gpu"
        self.n_gram_range = (1, 2)
        self.min_topic_size = 10
        self.diversity = 0
        self.nr_topics = "auto"

    def __repr__(self):
        return str(self.__dict__)


class BERTModel:
    def __init__(self, model: BERTopic, cfg: BERTConfig):
        self.model = model
        self.cfg = cfg


def train_bert(train_publications, conf: BERTConfig, debug_logging=True):
    sentence_model = SentenceTransformer(conf.embedding_model, device=conf.device)
    topic_model = BERTopic(embedding_model=sentence_model, verbose=debug_logging, n_gram_range=conf.n_gram_range,
                           min_topic_size=conf.min_topic_size, diversity=conf.diversity, nr_topics=conf.nr_topics)
    docs = train_publications['abstract_text_clean'].tolist()
    topic_model.fit_transform(docs)
    return BERTModel(topic_model, conf)


def save_bert_model(model: BERTModel):
    model.model.save("model.bert", save_embedding_model=True)
    common.save_pickle(model.cfg, "model.bert.cfg")


def load_bert_model():
    return BERTModel(
        BERTopic.load("model.bert"),
        common.load_pickle("model.bert.cfg"))


def eval_bert(model: BERTModel, publications: pd.DataFrame):
    features = model.model.transform(publications['abstract_text_clean'].tolist())
    publications = publications.copy()
    publications['feature'] = features
    return publications


def visualize_topics(model: BERTModel):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 4, figsize=(16, 16))
    i = 0
    for topic_name, topic in model.model.get_topics():
        wc = wordcloud.WordCloud(width=400, height=400)
        wc.generate_from_frequencies({token[0]: token[1] for token in topic})
        ax = axes[i // 4][i % 4]
        ax.imshow(wc.to_image(), interpolation='bilinear')
        ax.axis("off")
        i += 1
    while i < 12:
        ax = axes[i // 4][i % 4]
        ax.axis("off")
        i += 1
    fig.show()


def train_and_evaluate_bert(publications_train: pd.DataFrame, publications_cv: pd.DataFrame,
                            authors_train: pd.DataFrame, authors_cv: pd.DataFrame, users: pd.DataFrame,
                            conf: BERTConfig, debug_logging=False, save_model=False):
    pass
