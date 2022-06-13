import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import wordcloud


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


def normalize_array(v: np.ndarray):
    return v / np.linalg.norm(v)


def publications_for_user(publications: pd.DataFrame, authors: pd.DataFrame, user_id: int):
    publication_ids = authors[authors['user_id'] == user_id]['publication_id']
    return publications.loc[publication_ids]


def display_wordcloud(words_weights, figsize=(16, 16)):
    if type(words_weights) is dict:
        words_weights = [words_weights]
    if len(words_weights) == 1:
        wc = wordcloud.WordCloud(width=800, height=800)
        wc.generate_from_frequencies(words_weights[0])
        plt.imshow(wc.to_image(), interpolation='bilinear')
        plt.axis("off")
    else:
        plt.style.use('dark_background')
        rows = (len(words_weights) + 3) // 4
        fig, axes = plt.subplots(rows, 4, figsize=figsize)
        i = 0
        for words_dict in words_weights:
            wc = wordcloud.WordCloud(width=400, height=400)
            wc.generate_from_frequencies(words_dict)
            ax = axes[i // 4][i % 4]
            ax.imshow(wc.to_image(), interpolation='bilinear')
            ax.axis("off")
            i += 1
        while i < rows * 4:
            ax = axes[i // 4][i % 4]
            ax.axis("off")
            i += 1
        fig.show()
