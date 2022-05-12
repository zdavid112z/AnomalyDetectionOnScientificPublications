import pandas as pd
from sklearn.model_selection import train_test_split


class TrainConfig:
    def __init__(self, test_size=0.2, cv_size=0.2):
        self.test_size = test_size
        self.cv_size = cv_size


def split_train_cv_test_simple(publications: pd.DataFrame, conf: TrainConfig):
    train_cv, test = train_test_split(publications, test_size=conf.test_size, random_state=42)
    train, cv = train_test_split(train_cv, test_size=conf.cv_size * len(publications) / len(train_cv), shuffle=False)
    return train, cv, test


def split_train_cv_test_time_based(publications: pd.DataFrame, conf: TrainConfig):
    raise Exception("not implemented")


def split_authors_by_publications(publications_train: pd.DataFrame, publications_cv: pd.DataFrame,
                                  publications_test: pd.DataFrame, authors: pd.DataFrame):
    train_index = set(publications_train.index)
    cv_index = set(publications_cv.index)
    test_index = set(publications_test.index)

    train_mask = authors['publication_id'].progress_apply(lambda pub_id: pub_id in train_index)
    cv_mask = authors['publication_id'].progress_apply(lambda pub_id: pub_id in cv_index)
    test_mask = authors['publication_id'].progress_apply(lambda pub_id: pub_id in test_index)

    return authors[train_mask].copy(), authors[cv_mask].copy(), authors[test_mask].copy()
