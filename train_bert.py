import pandas as pd
import numpy as np
import common
import user_profile
import wordcloud
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import umap
from gensim.models.coherencemodel import CoherenceModel


sentence_models = {}


class BERTConfig:
    def __init__(self):
        self.embedding_model = "all-mpnet-base-v2"
        self.device = "cuda"
        self.n_components = 15
        self.batch_size = 128
        self.fpr_samples_from = 0
        self.fpr_samples_to = 1
        self.fpr_samples_count = 3000
        self.threshold = 0
        self.normalize_features = False
        self.threshold_overwrite = None
        self.metric = "cos"
        self.umap_metric = None
        self.umap_n_neighbors = 10
        self.umap_min_dist = 0
        self.reducer = "pca"
        self.verbose = False
        self.use_scaler = True

    def __repr__(self):
        return str(self.__dict__)


class BERTModel:
    def __init__(self, sentence_model: SentenceTransformer, scaler: StandardScaler, umap_reducer: umap.UMAP,
                 pca_reducer: PCA, cfg: BERTConfig):
        self.cfg = cfg
        self.sentence_model = sentence_model
        self.umap_reducer = umap_reducer
        self.pca_reducer = pca_reducer
        self.scaler = scaler
        self.reducer = None
        if self.umap_reducer is not None:
            self.reducer = umap_reducer
        if self.pca_reducer is not None:
            self.reducer = pca_reducer


def matrix_to_list(v):
    result = []
    for i in range(v.shape[0]):
        result.append(v[i, :])
    return result


def series_to_matrix(s):
    m = np.zeros((s.size, s.iloc[0].shape[0]))
    for i in range(s.size):
        m[i, :] = s.iloc[i]
    return m


def eval_bert_embeddings(publications: pd.DataFrame, sentence_model: SentenceTransformer, sentence_model_name: str,
                         batch_size: int, recalculate_embeddings=True, progress=True, normalize_embeddings=False):
    cached_embeddings = True
    if recalculate_embeddings:
        cached_embeddings = False
    elif 'embeddings' not in publications:
        cached_embeddings = False
    elif 'embeddings_sentence_model' not in publications:
        cached_embeddings = False
    elif any(publications['embeddings_sentence_model'] != sentence_model_name):
        cached_embeddings = False

    if not cached_embeddings:
        embeddings = sentence_model.encode(publications['abstract_text_clean'].tolist(),
                                           show_progress_bar=progress, batch_size=batch_size)
        embeddings_list = matrix_to_list(embeddings)
        if normalize_embeddings:
            embeddings_list = [common.normalize_array(v) for v in embeddings_list]
        publications['embeddings'] = pd.Series(embeddings_list, index=publications.index)
        publications['embeddings_sentence_model'] = sentence_model_name
    else:
        embeddings = series_to_matrix(publications['embeddings'])
    return embeddings


def train_bert(publications_train: pd.DataFrame, conf: BERTConfig, recalculate_embeddings=True, progress=True):
    publications_train = publications_train.copy()

    if conf.embedding_model in sentence_models:
        sentence_model = sentence_models[conf.embedding_model]
    else:
        sentence_model = SentenceTransformer(conf.embedding_model, device=conf.device)
        sentence_models[conf.embedding_model] = sentence_model

    embeddings = eval_bert_embeddings(publications_train, sentence_model, conf.embedding_model,
                                      recalculate_embeddings=recalculate_embeddings, progress=progress,
                                      batch_size=conf.batch_size,
                                      normalize_embeddings=conf.normalize_features)
    scaler = None
    if conf.use_scaler:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

    pca_reducer = None
    umap_reducer = None
    if conf.reducer == "pca":
        pca_reducer = PCA(n_components=conf.n_components)

        features = pca_reducer.fit_transform(embeddings)
    elif conf.reducer == "umap":
        if conf.umap_metric is None:
            conf.umap_metric = conf.metric
        umap_reducer = umap.UMAP(n_components=conf.n_components, metric=conf.umap_metric,
                                 n_neighbors=conf.umap_n_neighbors,
                                 min_dist=conf.umap_min_dist, verbose=conf.verbose)

        features = umap_reducer.fit_transform(embeddings)
    elif conf.reducer == 'none':
        features = embeddings
    else:
        raise Exception("reducer must be 'pca' or 'umap'")

    publications_train['feature'] = pd.Series(matrix_to_list(features), index=publications_train.index)
    if conf.normalize_features:
        publications_train['feature'] = publications_train['feature'].apply(common.normalize_array)
    return BERTModel(sentence_model, scaler, umap_reducer, pca_reducer, conf), publications_train


def save_bert_model(model: BERTModel):
    common.save_pickle(model.cfg, "model.bert.cfg")
    if model.scaler is not None:
        common.save_pickle(model.scaler, "model.bert.scaler")
    if model.cfg.reducer == "pca":
        common.save_pickle(model.pca_reducer, "model.bert.pca_reducer")
    elif model.cfg.reducer == "umap":
        common.save_pickle(model.umap_reducer, "model.bert.umap_reducer")


def load_bert_model():
    conf = common.load_pickle("model.bert.cfg")
    sentence_model = SentenceTransformer(conf.embedding_model, device=conf.device)
    scaler = None
    if conf.use_scaler:
        scaler = common.load_pickle("model.bert.scaler")
    pca_reducer = None
    umap_reducer = None
    if conf.reducer == "pca":
        pca_reducer = common.load_pickle("model.bert.pca_reducer")
    elif conf.reducer == "umap":
        umap_reducer = common.load_pickle("model.bert.umap_reducer")
    return BERTModel(sentence_model, scaler, umap_reducer, pca_reducer, conf)


def eval_bert(model: BERTModel, publications: pd.DataFrame, recalculate_embeddings=False, progress=True):
    publications = publications.copy()

    embeddings = eval_bert_embeddings(publications, model.sentence_model, model.cfg.embedding_model,
                                      recalculate_embeddings=recalculate_embeddings, progress=progress,
                                      batch_size=model.cfg.batch_size,
                                      normalize_embeddings=model.cfg.normalize_features)
    if model.cfg.use_scaler:
        embeddings_scaled = model.scaler.transform(embeddings)
    else:
        embeddings_scaled = embeddings

    if model.cfg.reducer == "pca":
        features = model.pca_reducer.transform(embeddings_scaled)
    elif model.cfg.reducer == "umap":
        features = model.umap_reducer.transform(embeddings_scaled)
    elif model.cfg.reducer == "none":
        features = embeddings_scaled
    else:
        raise Exception(f"unknown reducer {model.cfg.reducer}")
    publications['feature'] = pd.Series(matrix_to_list(features), index=publications.index)
    if model.cfg.normalize_features:
        publications['feature'] = publications['feature'].apply(common.normalize_array)
    return publications


def eval_features_for_words(model: BERTModel, publications: pd.DataFrame, ngram_range):
    if type(publications) == list:
        docs = [" ".join(p) for p in publications]
    elif type(publications) == pd.DataFrame:
        docs = publications['abstract_text_clean'].tolist()
    else:
        docs = [publications['abstract_text_clean']]
    vectorizer = CountVectorizer(stop_words='english', ngram_range=ngram_range)
    vectorizer.fit(docs)
    words = vectorizer.get_feature_names_out().tolist()
    word_embeddings = model.sentence_model.encode(words, batch_size=model.cfg.batch_size, show_progress_bar=True)
    return word_embeddings, words


def get_top_words(model: BERTModel, feature: np.ndarray, publications: pd.DataFrame = None,
                  word_embeddings=None, words=None, top_words=20,
                  ngram_range=(1, 3), metric='dot', figsize=(8, 8), plot=True):
    if model.reducer is not None:
        embedding_scaled = model.reducer.inverse_transform(feature)
    else:
        embedding_scaled = feature

    if model.scaler is not None:
        embedding = model.scaler.inverse_transform(embedding_scaled.reshape(1, -1))
    else:
        embedding = embedding_scaled

    if word_embeddings is None or words is None:
        word_embeddings, words = eval_features_for_words(model, publications, ngram_range)

    similarities = []
    for i in range(word_embeddings.shape[0]):
        similarity = user_profile.eval_score_simple(embedding, word_embeddings[i, :], metric)
        similarities.append(similarity)
    df = pd.DataFrame(similarities, index=words, columns=['similarity'])
    df = df.sort_values('similarity', ascending=False)

    best_words = df.head(top_words)
    best_words_weights = {}
    for word in best_words.index:
        best_words_weights[word] = best_words.loc[word]['similarity']
    if plot:
        common.display_wordcloud(best_words_weights, figsize=figsize)
    return best_words


def train_and_evaluate_bert(publications_train: pd.DataFrame, publications_cv: pd.DataFrame,
                            authors_train: pd.DataFrame, authors_cv: pd.DataFrame, authors_negative_cv: pd.DataFrame,
                            users: pd.DataFrame, conf: BERTConfig, save_model=False, plot=False,
                            random_negative_examples=True, recalculate_embeddings=False, progress=True,
                            figsize=(8, 8)):
    model, publications_train = train_bert(publications_train, conf,
                                           recalculate_embeddings=recalculate_embeddings, progress=progress)
    publications_cv = eval_bert(model, publications_cv, recalculate_embeddings=recalculate_embeddings,
                                progress=progress)

    fpr_samples = np.linspace(conf.fpr_samples_from, conf.fpr_samples_to, conf.fpr_samples_count)
    best_threshold, authors_cv, authors_negative_cv, users_features, performance_report = \
        user_profile.evaluate_and_fine_tune_model(publications_train, publications_cv, authors_train, authors_cv,
                                                  authors_negative_cv, users, metric=conf.metric,
                                                  random_negative_examples=random_negative_examples,
                                                  fpr_samples=fpr_samples, plot=plot, figsize=figsize,
                                                  threshold_overwrite=conf.threshold_overwrite)

    model.cfg.threshold = best_threshold
    if save_model:
        save_bert_model(model)

    return model, publications_train, publications_cv, authors_cv, authors_negative_cv, users_features, \
        performance_report | conf.__dict__
