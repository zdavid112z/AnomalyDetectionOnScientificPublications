import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, fbeta_score
import seaborn as sns

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
    return np.mean(np.array(stds))


def fill_zero_std_with(users_features: pd.DataFrame, default_std: float):
    users_features = users_features.copy()
    users_features['features_std'] = users_features['features_std'].apply(
        lambda stds: np.array([std if std >= EPS else default_std for std in stds]))
    return users_features


def log_norm(x, mean, std):
    return -np.power((x - mean) / std, 2) / 2 - np.log(std * np.sqrt(2 * np.pi))


def eval_topics_scores(publications: pd.DataFrame, users_features: pd.DataFrame, authors: pd.DataFrame, progress=False):
    def eval_author(author):
        try:
            pub_id = int(author['publication_id'])
            user_id = int(author['user_id'])
            pub = publications.loc[pub_id]
            feature = pub['feature']
            user = users_features.loc[user_id]
            mean = user['features_mean']
            std = user['features_std']
        except Exception as e:
            return None
        return log_norm(feature, mean, std)

    authors = authors.copy()
    if progress:
        authors['topics_scores'] = authors.progress_apply(eval_author, axis=1)
    else:
        authors['topics_scores'] = authors.apply(eval_author, axis=1)
    authors = authors.dropna(subset='topics_scores')
    authors['score'] = authors['topics_scores'].apply(np.sum)
    return authors


def eval_topics_scores_random(publications_sample: pd.DataFrame, users_features_sample: pd.DataFrame):
    n = len(publications_sample)
    scores = np.zeros(n)
    for i in range(n):
        pub = publications_sample.iloc[i]
        user = users_features_sample.iloc[i]
        feature = pub['feature']
        mean = user['features_mean']
        std = user['features_std']
        s = log_norm(feature, mean, std)
        scores[i] = np.sum(s)
    return pd.Series(scores.tolist())


def plot_roc_curve(positive_author_scores:pd.Series, negative_author_scores:pd.Series, fpr_samples=np.linspace(0.04, 0.3, 53), plot=True):
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
        plt.figure(figsize=(16, 16))
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


def plot_fbeta_plot(positive_author_scores: pd.Series, negative_author_scores: pd.Series, beta=1, thresholds=None, plot=True):
    y_test = np.concatenate((np.ones(len(positive_author_scores)), np.zeros(len(negative_author_scores))))
    model_probs = np.concatenate((positive_author_scores.to_numpy(), negative_author_scores.to_numpy()))
    if thresholds is None:
        thresholds = sorted(model_probs.tolist())
    fbeta_scores = []
    best_fbeta_score = -1
    best_threshold = 0
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        # score = fbeta_score(y_test, model_probs > threshold, beta=beta)
        cf_matrix = get_confusion_matrix(positive_author_scores, negative_author_scores, threshold)
        if cf_matrix[1][1] == 0:
            score = 0
        else:
            precision = cf_matrix[1][1] / (cf_matrix[1][1] + cf_matrix[0][1])
            recall = cf_matrix[1][1] / (cf_matrix[1][1] + cf_matrix[1][0])
            score = 2 * precision * recall / (precision + recall)
        fbeta_scores.append(score)
        if best_fbeta_score < score:
            best_fbeta_score = score
            best_threshold = threshold
    if plot:
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111)
        plt.plot(thresholds, fbeta_scores)
        ax.set_title("F1 score by Threshold")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1 score")
    return best_fbeta_score, best_threshold


def get_confusion_matrix(positive_author_scores: pd.Series, negative_author_scores: pd.Series, threshold):
    true_positives = len(positive_author_scores[positive_author_scores > threshold])
    false_positives = len(positive_author_scores) - true_positives
    false_negatives = len(negative_author_scores[negative_author_scores > threshold])
    true_negatives = len(negative_author_scores) - false_negatives
    return np.array([[true_negatives, false_positives], [false_negatives, true_positives]])


def plot_confusion_matrix(positive_author_scores:pd.Series, negative_author_scores:pd.Series, threshold):
    cf_matrix = get_confusion_matrix(positive_author_scores, negative_author_scores, threshold)

    plt.figure(figsize=(16, 16))
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


def get_precision(cf_matrix):
    return (cf_matrix[0][0] + cf_matrix[1][1]) / (cf_matrix[0][0] + cf_matrix[0][1] + cf_matrix[1][0] + cf_matrix[1][1])


def get_negative_scores(model, publications_cv, users_features, num_negative_examples):
    rng = np.random.default_rng()
    publications_sample = rng.integers(0, len(publications_cv), num_negative_examples)
    users_sample = rng.integers(0, len(users_features), num_negative_examples)
    return eval_topics_scores_random(publications_cv.iloc[publications_sample], users_features.iloc[users_sample])
