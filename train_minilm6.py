import matplotlib.pyplot as plt
import common
import preprocess_common
import preprocess_lda
import train_common
import train_lda
import user_profile
import preprocess_bert
import train_bert

from pprint import pprint
import pandas as pd
import numpy as np
import itertools
import time
from tqdm import tqdm
tqdm.pandas()

publications, users, authors_raw = common.load_raw_datasets()
authors = authors_raw[authors_raw['state'] == 'validatAcceptat']
authors_negative = authors_raw[authors_raw['state'] == 'validatRefuzat']
common_dataset_cached = True
lda_dataset_cached = True
bert_dataset_cached = True
lda_model_cached = True
bert_model_cached = True
lda_visualize_results = False
lda_visualize_test_results = False
bert_features_cached = True

split_cfg = train_common.TrainConfig(cv_size=0.2, test_size=0.2)

if not common_dataset_cached:
    nlp = preprocess_common.init_nlp()
    publications_en = preprocess_common.preprocess_publications_common(publications, nlp)
else:
    publications_en = common.load_dataframe("publications_en")

if not bert_dataset_cached:
    publications_bert = preprocess_bert.preprocess_bert(publications_en)
else:
    publications_bert = common.load_dataframe("publications_bert")

publications_bert_train, publications_bert_cv, publications_bert_test = \
    train_common.split_train_cv_test_simple(publications_bert, split_cfg)

authors_bert_train, authors_bert_cv, authors_bert_test = \
    train_common.split_authors_by_publications(
        publications_bert_train, publications_bert_cv, publications_bert_test, authors)

authors_negative_bert_train, authors_negative_bert_cv, authors_negative_bert_test = \
    train_common.split_authors_by_publications(
        publications_bert_train, publications_bert_cv, publications_bert_test, authors_negative)

if not bert_model_cached:
    bert_conf = train_bert.BERTConfig()
    bert_conf.embedding_model = "all-MiniLM-L6-v2"
    bert_conf.device = "cuda"
    bert_conf.pca_num_components = 16
    bert_conf.batch_size = 32
    bert_conf.fpr_samples_from = 0
    bert_conf.fpr_samples_to = 1
    bert_conf.fpr_samples_count = 3000
    bert_conf.normalize_features = False
    bert_conf.metric = "dot"

    bert, publications_bert_train, publications_bert_cv, authors_bert_cv, \
    authors_negative_bert_cv, users_features, performance_report = \
        train_bert.train_and_evaluate_bert(
            publications_bert_train, publications_bert_cv, authors_bert_train, authors_bert_cv,
            authors_negative_bert_cv, users, bert_conf, save_model=True, plot=True,
            random_negative_examples=False, recalculate_embeddings=False, progress=True,
            figsize=(8, 8))
    print(performance_report)

    publications_bert_test = train_bert.eval_bert(bert, publications_bert_test,
                                                  recalculate_embeddings=False, progress=True)

    common.save_dataframe(publications_bert_train, "publications_bert_train_minilm6")
    common.save_dataframe(publications_bert_cv, "publications_bert_cv_minilm6")
    common.save_dataframe(publications_bert_test, "publications_bert_test_minilm6")
else:
    publications_bert_train = common.load_dataframe("publications_bert_train_minilm6")
    publications_bert_cv = common.load_dataframe("publications_bert_cv_minilm6")
    publications_bert_test = common.load_dataframe("publications_bert_test_minilm6")
    bert = train_bert.load_bert_model()

pca_num_components = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 35, 50, 75, 100]
normalize_features = [False, True]
metric = ['cos']
choices = list(itertools.product(pca_num_components, normalize_features, metric))
np.random.shuffle(choices)
# results = common.load_pickle("results_minilm6_1655467361_48.pickle")
results = []
filename = f"results_minilm6_{int(time.time())}"
i = 0
for pca_num_components, normalize_features, metric in tqdm(choices):
    duplicate = False
    for r in results:
        if r['pca_num_components'] == pca_num_components and r['normalize_features'] == normalize_features and r['metric'] == metric:
            duplicate = True
            break
    if duplicate:
        continue

    bert_conf = train_bert.BERTConfig()
    bert_conf.embedding_model = "all-MiniLM-L6-v2"
    bert_conf.device = "cuda"
    bert_conf.pca_num_components = pca_num_components
    bert_conf.batch_size = 32
    bert_conf.fpr_samples_from = 0
    bert_conf.fpr_samples_to = 1
    bert_conf.fpr_samples_count = 2
    bert_conf.normalize_features = normalize_features
    bert_conf.metric = metric

    _, _, _, _, _, _, performance_report = \
        train_bert.train_and_evaluate_bert(
            publications_bert_train, publications_bert_cv, authors_bert_train, authors_bert_cv,
            authors_negative_bert_cv, users, bert_conf, save_model=False, plot=False,
            random_negative_examples=False, recalculate_embeddings=False, progress=False,
            threshold_overwrite=0)

    results.append(performance_report)
    print(performance_report)
    common.save_pickle(results, f"{filename}_{i}.pickle")
    i += 1

print(results)
