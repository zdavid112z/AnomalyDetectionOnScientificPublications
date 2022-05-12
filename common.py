import pandas as pd
import pickle
import os


def load_raw_datasets():
    publications_filename = 'publications_202107181309.csv'
    users_filename = 'users_202107181309.csv'
    authors_filename = 'authors_202107181309.csv'
    publications, users, authors = pd.read_csv(publications_filename), pd.read_csv(users_filename), pd.read_csv(
        authors_filename)
    publications = publications.set_index('id')
    users = users.set_index('id')
    authors = authors.set_index('id')
    return publications, users, authors


def save_dataframe(df: pd.DataFrame, name: str):
    df.to_hdf(f'{name}.hdf', key='df', mode='w')


def load_dataframe(name: str):
    try:
        return pd.read_hdf(f"{name}.hdf", "df")
    except Exception as e:
        print(f"Warning! Failed to load dataframe: {e}")
        return None


def make_sorted_dict(m):
    s = sorted(m.items(), key=lambda x: -x[1])
    return dict(s)


def save_pickle(obj, name: str):
    filename = f"{name}.pickle"
    file_to_store = open(filename, "wb", 0)
    pickle.dump(obj, file_to_store)
    file_to_store.close()


def load_pickle(name: str):
    filename = f"{name}.pickle"
    file_to_read = open(filename, "rb")
    obj = pickle.load(file_to_read)
    file_to_read.close()
    return obj
