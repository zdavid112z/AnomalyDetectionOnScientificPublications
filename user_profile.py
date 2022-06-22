import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial.distance
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, fbeta_score
import seaborn as sns
from sklearn import metrics
from numpy import dot
from numpy.linalg import norm
from scipy.spatial.distance import cosine

import common

EPS = 0.0001


def add_publication_features_to_users(publications: pd.DataFrame, users: pd.DataFrame, authors: pd.DataFrame):
    new_users = users.copy()

    def add_feature(author):
        try:
            pub_id = int(author['publication_id'])
            user_id = int(author['user_id'])
            pub = publications.loc[pub_id]
            feature = pub['feature']
        except:
            return
        new_users['publication_features'][user_id].append(feature)
        new_users['publication_ids'][user_id].append(pub_id)

    new_users['publication_features'] = np.empty((len(users), 0)).tolist()
    new_users['publication_ids'] = np.empty((len(users), 0)).tolist()
    authors.apply(add_feature, axis=1)
    return new_users


def build_user_profiles_simple(users_features: pd.DataFrame):
    users_features['features_mean'] = users_features['publication_features'].apply(
        lambda features: np.mean(np.asarray(features), axis=0) if len(features) > 0 else None)
    users_features['features_std'] = users_features['publication_features'].apply(
        lambda features: np.std(np.asarray(features), axis=0) if len(features) > 0 else None)
    users_features['profile'] = users_features['features_mean']
    return users_features


def median_nonzero_std(users_features: pd.DataFrame):
    stds = []
    users_features['features_std'].apply(lambda v: [stds.append(x) for x in v.tolist() if x >= EPS])
    return np.median(np.array(stds))


def mean_nonzero_std(users_features: pd.DataFrame):
    stds = []
    users_features['features_std'].apply(lambda v: [stds.append(x) for x in v.tolist() if x >= EPS])
    return float(np.mean(np.array(stds)))


def fill_zero_std_with(users_features: pd.DataFrame, default_std: float):
    users_features = users_features.copy()
    users_features['features_std'] = users_features['features_std'].apply(
        lambda stds: np.array([std if std >= EPS else default_std for std in stds]))
    return users_features


def log_norm(x, mean, std):
    return -np.power((x - mean) / std, 2) / 2 - np.log(std * np.sqrt(2 * np.pi))


def cos_similarity(a, b):
    return np.dot(a, b) / (np.norm(a) * np.norm(b))


# Bigger is better
def eval_score_simple(a, b, metric):
    if metric == "cos" or metric == "cosine" or metric == "cos_max" or metric == "cosine_max":
        return (np.dot(a, b) / np.linalg.norm(a)) / np.linalg.norm(b)
        # return 1-scipy.spatial.distance.cosine(a, b)
    elif metric == "dot":
        return np.dot(a, b)
    elif metric == "norm":
        return -np.linalg.norm(a - b)
    elif metric == "euclidean" or metric == "euclidean_max":
        return -np.linalg.norm(a - b)
    raise Exception("metric must be 'cos', 'dot', 'norm' or 'euclidean")


# Bigger is better
def eval_score(feature, mean, std, features, metric):
    if metric == "cos" or metric == "cosine":
        return (np.dot(feature, mean) / np.linalg.norm(feature)) / np.linalg.norm(mean)
        # return 1 - scipy.spatial.distance.cosine(feature, mean)
    elif metric == "dot":
        return np.dot(feature, mean)
    elif metric == "norm":
        return log_norm(feature, mean, std)
    elif metric == "euclidean":
        return -np.linalg.norm(feature - mean)
    elif metric == "cos_max" or metric == "cosine_max":
        max_score = None
        for f in features:
            # score = (np.dot(feature, f) / np.linalg.norm(feature)) / np.linalg.norm(f)
            score = scipy.spatial.distance.cosine(feature, f)
            if max_score is None or score > max_score:
                max_score = score
        return max_score
    elif metric == "euclidean_max":
        max_score = None
        for f in features:
            score = -np.linalg.norm(feature - f)
            if max_score is None or score > max_score:
                max_score = score
        return max_score
    raise Exception("metric must be 'cos', 'dot', 'norm', 'euclidean', 'cos_max' or 'euclidean_max'")


def eval_topics_scores(publications: pd.DataFrame, users_features: pd.DataFrame, authors: pd.DataFrame, metric: str,
                       progress=False):
    def eval_author(author):
        try:
            pub_id = int(author['publication_id'])
            user_id = int(author['user_id'])
            pub = publications.loc[pub_id]
            feature = pub['feature']
            user = users_features.loc[user_id]
            features = user['publication_features']
            mean = user['features_mean']
            std = user['features_std']
        except Exception as e:
            return None
        return eval_score(feature, mean, std, features, metric)

    authors = authors.copy()
    if progress:
        authors['topics_scores'] = authors.progress_apply(eval_author, axis=1)
    else:
        authors['topics_scores'] = authors.apply(eval_author, axis=1)
    authors = authors.dropna(subset='topics_scores')
    authors['score'] = authors['topics_scores'].apply(np.sum)
    return authors


def eval_topics_scores_random(publications_sample: pd.DataFrame, users_features_sample: pd.DataFrame, metric):
    n = len(publications_sample)
    scores = np.zeros(n)
    for i in range(n):
        pub = publications_sample.iloc[i]
        user = users_features_sample.iloc[i]
        feature = pub['feature']
        features = user['publication_features']
        mean = user['features_mean']
        std = user['features_std']
        s = eval_score(feature, mean, std, features, metric)
        scores[i] = np.sum(s)
    return pd.Series(scores.tolist())


def plot_roc_curve(positive_author_scores: pd.Series, negative_author_scores: pd.Series,
                   fpr_samples=np.linspace(0.04, 0.3, 53), plot=True, figsize=(16, 16)):
    y_test = np.concatenate((np.ones(len(positive_author_scores)), np.zeros(len(negative_author_scores))))
    model_probs = np.concatenate((positive_author_scores.to_numpy(), negative_author_scores.to_numpy()))
    random_probs = [0 for _ in range(len(y_test))]
    # calculate AUC
    model_auc = roc_auc_score(y_test, model_probs)
    # summarize score
    if plot:
        print(f'Model: ROC AUC={model_auc}')
    # calculate ROC Curve
    # For the Random Model
    random_fpr, random_tpr, _ = roc_curve(y_test, random_probs)
    # For the actual model
    model_fpr, model_tpr, thresholds = roc_curve(y_test, model_probs)
    # Plot the roc curve for the model and the random model line
    if plot:
        plt.style.use('dark_background')
        plt.figure(figsize=figsize)
        plt.plot(random_fpr, random_tpr, linestyle='--', label='Random')
        plt.plot(model_fpr, model_tpr, marker='.', label=f'Model (AUC={format(model_auc, ".2f")})')
        # Create labels for the axis
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("ROC Curve")
        # show the legend
        plt.legend()

    fpr_index = 0
    fpr_to_threshold = pd.DataFrame(columns=["fpr", "threshold"])
    fpr_to_threshold = fpr_to_threshold.set_index('fpr')
    for i in range(len(model_fpr)):
        if fpr_index < len(fpr_samples) and fpr_samples[fpr_index] <= model_fpr[i]:
            fpr_to_threshold.loc[fpr_samples[fpr_index]] = thresholds[i]
            fpr_index += 1

    return model_auc, fpr_to_threshold, thresholds


def plot_score(positive_author_scores: pd.Series, negative_author_scores: pd.Series, thresholds=None,
               plot=True, figsize=(16, 16), score="phi"):
    y_test = np.concatenate((np.ones(len(positive_author_scores)), np.zeros(len(negative_author_scores))))
    model_probs = np.concatenate((positive_author_scores.to_numpy(), negative_author_scores.to_numpy()))
    if thresholds is None:
        thresholds = sorted(model_probs.tolist())
    fbeta_scores = []
    best_score = -1
    best_threshold = 0
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        # score = fbeta_score(y_test, model_probs > threshold, beta=beta)
        tn, fp, fn, tp = get_confusion_matrix(positive_author_scores, negative_author_scores, threshold).ravel()
        # if cf_matrix[1][1] == 0:
        #     score = 0
        # else:
        #     precision = cf_matrix[1][1] / (cf_matrix[1][1] + cf_matrix[0][1])
        #     recall = cf_matrix[1][1] / (cf_matrix[1][1] + cf_matrix[1][0])
        #     score = 2 * precision * recall / (precision + recall)
        if score == "phi":
            if (tp == 0 or tn == 0) and (fp == 0 or fn == 0):
                s = 0
            else:
                s = (tp * tn - fn * fp) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        elif score == "f1":
            s = get_f1_score(tn, fp, fn, tp)
        else:
            raise Exception("Unknown score")
        fbeta_scores.append(s)
        if best_score < s:
            best_score = s
            best_threshold = threshold
    if plot:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        plt.plot(thresholds, fbeta_scores)
        ax.set_title(f"{score.upper()} score by Threshold")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(f"{score.upper()} score")
    return best_score, best_threshold


def get_confusion_matrix(positive_author_scores: pd.Series, negative_author_scores: pd.Series, threshold):
    y_actual = np.concatenate((np.ones(len(positive_author_scores)), np.zeros(len(negative_author_scores))))
    y_predicted = np.concatenate((positive_author_scores > threshold, negative_author_scores > threshold))
    return metrics.confusion_matrix(y_actual, y_predicted)


def plot_confusion_matrix(positive_author_scores: pd.Series, negative_author_scores: pd.Series, threshold,
                          figsize=(16, 16)):
    cf_matrix = get_confusion_matrix(positive_author_scores, negative_author_scores, threshold)

    plt.figure(figsize=figsize)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.show()


def get_precision(tn, fp, fn, tp):
    return (tp + tn) / (tp + tn + fn + fp)


def get_f1_score(tn, fp, fn, tp):
    if tp == 0:
        return 0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    return f1


def get_negative_scores(publications_cv, users_features, num_negative_examples, metric):
    rng = np.random.default_rng()
    publications_sample = rng.integers(0, len(publications_cv), num_negative_examples)
    users_sample = rng.integers(0, len(users_features), num_negative_examples)
    return eval_topics_scores_random(publications_cv.iloc[publications_sample], users_features.iloc[users_sample],
                                     metric=metric)


def evaluate_and_fine_tune_model(publications_train: pd.DataFrame, publications_cv: pd.DataFrame,
                                 authors_train: pd.DataFrame, authors_cv: pd.DataFrame,
                                 authors_negative_cv: pd.DataFrame, users: pd.DataFrame,
                                 metric: str, random_negative_examples: bool, fpr_samples: np.ndarray,
                                 plot: bool, figsize=(8, 8), threshold_overwrite=None,
                                 score="phi"):
    users_features = add_publication_features_to_users(publications_train, users, authors_train)
    users_features = build_user_profiles_simple(users_features)
    users_features = users_features.dropna(subset='profile')
    users_features = fill_zero_std_with(users_features, mean_nonzero_std(users_features))

    authors_cv = eval_topics_scores(publications_cv, users_features, authors_cv, metric=metric)
    positive_scores = authors_cv['score']

    authors_negative_cv = eval_topics_scores(publications_cv, users_features, authors_negative_cv, metric=metric)
    if random_negative_examples:
        negative_scores = get_negative_scores(publications_cv, users_features, len(authors_cv), metric=metric)
    else:
        negative_scores = authors_negative_cv['score']

    auc, fpr_to_threshold, _ = plot_roc_curve(positive_scores, negative_scores, plot=plot, fpr_samples=fpr_samples,
                                              figsize=figsize)
    phi_score, best_threshold = plot_score(positive_scores, negative_scores, plot=plot, figsize=figsize,
                                           thresholds=fpr_to_threshold['threshold'].tolist(),
                                           score=score)
    if threshold_overwrite is not None:
        best_threshold = threshold_overwrite
    if plot:
        plot_confusion_matrix(positive_scores, negative_scores, best_threshold, figsize=figsize)

    tn, fp, fn, tp = get_confusion_matrix(positive_scores, negative_scores, best_threshold).ravel()
    precision = get_precision(tn, fp, fn, tp)
    f1_score = get_f1_score(tn, fp, fn, tp)

    return best_threshold, authors_cv, authors_negative_cv, users_features, {
        "auc": auc,
        "phi_score": phi_score,
        "f1_score": f1_score,
        "precision": precision,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
