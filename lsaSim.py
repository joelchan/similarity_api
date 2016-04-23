from gensim import corpora, models, similarities
import pandas as pd

dictionary = corpora.Dictionary.load("data/dictionary_fabric")
lsi = models.LsiModel.load("data/lsi_300_fabric")
paths = pd.read_csv("data/fabric_paths.csv")

def rank_paths(ideaBag):
    """Given a set of ideas, return a ranking of solution paths
    in terms of similarity to the set of ideas.
    """

    # create similarity index from paths
    paths['allwords'] = [p.encode('utf-8', 'ignore') for p in paths['allwords']] # deal with unicode
    path_corpus = [dictionary.doc2bow(p.split()) for p in paths['allwords']] # create the corpus
    index = similarities.MatrixSimilarity(lsi[path_corpus]) # create the index

    # preprocess and project into lsi space
    vec_bow = dictionary.doc2bow(ideaBag.lower().split())
    vec_lsi = lsi[vec_bow]

    # get similarities
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    # map similarities to path data
    paths['rank'] = 0
    paths['sim'] = 0.0
    for rank, docSim in enumerate(sims):
        paths.set_value(docSim[0], 'rank', rank)
        paths.set_value(docSim[0], 'sim', docSim[1])

    paths.sort_values("rank", inplace=True)

    return paths[['id', 'path', 'rank', 'sim']].to_json(orient="records")
