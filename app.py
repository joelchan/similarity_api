from flask import Flask, jsonify, request
from gensim import corpora, models, similarities
import pandas as pd
import logging

app = Flask(__name__)

"""
Routes
"""
# use this to test if the API is live
@app.route('/')
def hello():
    return "Hello World!"

# given a set of ideas and a knowledge test, return the 
@app.route('/LSArank', methods=['GET', 'POST'])
def get_sim_ranks():
    data = request.get_json()
    ideas = data['ideas']
    alignType = data['alignType']
    kTest = data['kTest']
    ideaBag = " ".join(ideas)
    rankings = rank_paths(ideaBag, kTest)
    app.logger.info(rankings)
    selection = select_path(rankings, alignType)
    # app.logger.info(selection)
    rankings_simple = {}
    for index, row in rankings.iterrows():
        rankings_simple[row['knowledgeBase']] = {'rank': row['rank'], 'sim': row['sim']}
    return jsonify(selection=selection,
        rankings=rankings_simple)

"""
Helper functions
"""

dictionary = corpora.Dictionary.load("data/dictionary_fabric")
lsi = models.LsiModel.load("data/lsi_300_fabric")
paths = pd.read_csv("data/fabric_paths.csv")

def rank_paths(ideaBag, kTest):
    """Given a set of ideas, return a ranking of solution paths
    that is jointly determined by knowledge test scores
    and similarity to the set of ideas.
    """

    app.logger.info(ideaBag)

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
    paths['rank'] = 0.0
    for docSim in sims:
        app.logger.info(paths.loc[docSim[0], 'path'])
        kBase = paths.loc[docSim[0], 'knowledgeBase'] 
        app.logger.info(kTest)
        app.logger.info(kBase)
        # app.logger.info(type(kTest[kBase]))
        kBaseRank = int(kTest[kBase]) + docSim[1] # combine the knowledge test score and similarity score
        paths.set_value(docSim[0], 'rank', kBaseRank) 

        paths.set_value(docSim[0], 'sim', docSim[1]) # store the similarity to ideas

    # paths['rank'] = 0
    # paths['sim'] = 0.0
    # for rank, docSim in enumerate(sims):
        # paths.set_value(docSim[0], 'rank', rank)
        # paths.set_value(docSim[0], 'sim', docSim[1])

    paths.sort_values("rank", inplace=True, ascending=False)

    return paths[['id', 'path', 'rank', 'sim', 'knowledgeBase']]

def select_path(paths, alignType):
    if alignType == "m": # misaligned condition, so choose the lowest ranking one
        paths.sort_values("rank", inplace=True, ascending=True)
    app.logger.info(paths['knowledgeBase'])
    return paths['knowledgeBase'].iloc[0]

"""
Main function
"""
if __name__ == '__main__':
    app.run(debug=True)