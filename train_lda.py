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


class LDAConfig:
    def __init__(self):
        self.num_topics = 10
        self.chunksize = 5000
        self.passes = 4
        self.iterations = 400
        self.eval_every = 1
        self.no_below = 20
        self.no_above = 0.5
        self.fpr_samples_from = 0
        self.fpr_samples_to = 0.4
        self.fpr_samples_count = 161
        self.fpr_samples_count = 161
        self.num_negative_examples_to_positive_examples = 1
        self.threshold = None

    def __repr__(self):
        return str(self.__dict__)


class LDAModel:
    def __init__(self, model, dictionary, corpus, cfg: LDAConfig):
        self.model = model
        self.dictionary = dictionary
        self.corpus = corpus
        self.cfg = cfg


def train_lda(train_publications, conf, debug_logging=True):
    if debug_logging:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG, force=True)
    else:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARN, force=True)
    corpus_unsplit = train_publications['abstract_text_clean'].tolist()
    corpus = [text.split(' ') for text in corpus_unsplit]
    dictionary = corpora.Dictionary(corpus)
    dictionary.filter_extremes(no_below=conf.no_below, no_above=conf.no_above)
    corpus_filtered = [dictionary.doc2bow(doc) for doc in corpus]
    if debug_logging:
        print('Number of unique tokens: %d' % len(dictionary))
        print('Number of documents: %d' % len(corpus))
    lda_model = gensim.models.LdaModel(corpus=corpus_filtered,
                                       id2word=dictionary,
                                       chunksize=conf.chunksize,
                                       alpha='auto',
                                       eta='auto',
                                       iterations=conf.iterations,
                                       passes=conf.passes,
                                       eval_every=conf.eval_every,
                                       num_topics=conf.num_topics)
    top_topics = lda_model.top_topics(corpus_filtered)  # , num_words=20)
    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / conf.num_topics
    if debug_logging:
        print('Average topic coherence: %.4f.' % avg_topic_coherence)
        pprint(top_topics)
    return LDAModel(lda_model, dictionary, corpus_filtered, conf)


def save_lda_model(model: LDAModel):
    model.model.save("lda.model")
    model.dictionary.save("lda.dictionary")
    common.save_pickle(model.cfg, "lda_cfg")
    common.save_pickle(model.corpus, "lda_corpus")


def load_lda_model():
    return LDAModel(
        gensim.models.LdaModel.load("lda.model"),
        corpora.Dictionary.load("lda.dictionary"),
        common.load_pickle("lda_corpus"),
        common.load_pickle("lda_cfg"))


def eval_lda(model: LDAModel, publications: pd.DataFrame, progress=False):
    def get_feature(text):
        feature_compact = model.model[model.dictionary.doc2bow(text.split(' '))]
        feature = np.zeros(model.cfg.num_topics)
        for idx, value in feature_compact:
            feature[idx] = value
        return feature

    publications = publications.copy()
    if progress:
        publications['feature'] = publications['abstract_text_clean'].progress_apply(get_feature)
    else:
        publications['feature'] = publications['abstract_text_clean'].apply(get_feature)
    return publications


def visualize_topics(model: LDAModel):
    top_topics = model.model.top_topics(model.corpus)  # , num_words=20)
    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / model.cfg.num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 4, figsize=(16, 16))
    i = 0
    for topic in top_topics:
        wc = wordcloud.WordCloud(width=400, height=400)
        wc.generate_from_frequencies({token[1]: token[0] for token in topic[0]})
        ax = axes[i // 4][i % 4]
        ax.imshow(wc.to_image(), interpolation='bilinear')
        ax.axis("off")
        i += 1
    while i < 12:
        ax = axes[i // 4][i % 4]
        ax.axis("off")
        i += 1
    fig.show()


def train_and_evaluate_lda(publications_train: pd.DataFrame, publications_cv: pd.DataFrame, authors_train: pd.DataFrame,
                           authors_cv: pd.DataFrame, authors_negative_cv: pd.DataFrame, users: pd.DataFrame,
                           conf: LDAConfig, debug_logging=False, save_model=False, plot=False,
                           random_negative_examples=True):

    model = train_lda(publications_train, conf, debug_logging=debug_logging)
    publications_train = eval_lda(model, publications_train)

    users_features = user_profile.add_publication_features_to_users(publications_train, users, authors_train)
    users_features = user_profile.build_user_profiles_simple(users_features)
    users_features = users_features.dropna(subset='profile')
    users_features = user_profile.fill_zero_std_with(users_features, user_profile.mean_nonzero_std(users_features))

    publications_cv = eval_lda(model, publications_cv)
    authors_cv = user_profile.eval_topics_scores(publications_cv, users_features, authors_cv)
    positive_scores = authors_cv['score']

    if random_negative_examples:
        negative_scores = user_profile.get_negative_scores(model, publications_cv, users_features, len(authors_cv))
    else:
        authors_negative_cv = user_profile.eval_topics_scores(publications_cv, users_features, authors_negative_cv)
        negative_scores = authors_negative_cv['score']

    fpr_samples = np.linspace(conf.fpr_samples_from, conf.fpr_samples_to, conf.fpr_samples_count)
    auc, fpr_to_threshold, _ = user_profile.plot_roc_curve(positive_scores, negative_scores,
                                                           plot=plot, fpr_samples=fpr_samples)
    f1_score, best_threshold = user_profile.plot_fbeta_plot(positive_scores, negative_scores, plot=plot,
                                                            thresholds=fpr_to_threshold['threshold'].tolist())
    cf_matrix = user_profile.get_confusion_matrix(positive_scores, negative_scores, best_threshold)
    precision = user_profile.get_precision(cf_matrix)
    model.cfg.threshold = best_threshold
    if save_model:
        save_lda_model(model)

    return model, publications_train, publications_cv, authors_cv, authors_negative_cv, users_features, {
        "auc": auc,
        "f1_score": f1_score,
        "precision": precision,
        "cf_matrix": cf_matrix
    } | conf.__dict__
