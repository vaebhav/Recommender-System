#!/usr/local/bin/python3

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from scipy.stats.stats import pearsonr
from sklearn.metrics import jaccard_similarity_score
from sklearn.preprocessing import normalize


Ratings=pd.read_csv("/Users/vaebhav/Documents/Python/Machine Learning/Recommender System/Content Based/ratings.csv",encoding='utf8')
Movies= pd.read_csv("/Users/vaebhav/Documents/Python/Machine Learning/Recommender System/Content Based/movies.csv",encoding='utf8')
Tags=pd.read_csv("/Users/vaebhav/Documents/Python/Machine Learning/Recommender System/Content Based/tags.csv")



pivottb = pd.pivot_table(Ratings,index='userId',values='rating',columns='movieId',fill_value=0)

#Magnitude for eac ratings given by each user to normalize the individual rating vector

magnitude = np.sqrt(np.square(pivottb).sum(axis=1))

pivottb_mod = pivottb.divide(magnitude, axis='index')

def calculate_similarity(data_items):
    """Calculate the column-wise cosine similarity for a sparse
    matrix. Return a new dataframe matrix with similarities.
    """
    similarities = cosine_similarity(data_items.transpose())

    sim = pd.DataFrame(data=similarities, index= data_items.columns, columns= data_items.columns)
    return sim

data_matrix = calculate_similarity(pivottb_mod)

user = 316
user_index = Ratings[Ratings.userId == user].index.tolist()[0]

known_user_likes = pivottb.iloc[pivottb.index.get_loc(user)]
known_user_likes = known_user_likes[known_user_likes > 0].index.values

user_vector = pivottb.iloc[pivottb.index.get_loc(user)]

# Calculate the score.
score = data_matrix.dot(user_vector).div(data_matrix.sum(axis=1))

# Drop the known likes.
score = score.drop(known_user_likes)

#print (known_user_likes)

# Recommended movies based on movies similar movies and its similarity score for user 316
print (score.nlargest(20))
