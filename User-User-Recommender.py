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
import itertools
from pprint import pprints
from sklearn.metrics import pairwise_distances
import copy


Ratings=pd.read_csv("/Users/vaebhav/Documents/Python/Machine Learning/Recommender System/Content Based/ratings.csv",encoding='utf8')
Movies= pd.read_csv("/Users/vaebhav/Documents/Python/Machine Learning/Recommender System/Content Based/movies.csv",encoding='utf8')
Tags=pd.read_csv("/Users/vaebhav/Documents/Python/Machine Learning/Recommender System/Content Based/tags.csv")

#Dropping Timestamp
Ratings.drop(['timestamp'], inplace = True, axis = 1 )


# Compute Mean for non Zero column values across axis = 1

def computeMean(dataframe):
    sum = dataframe.sum()
    count = np.count_nonzero(dataframe)
    return sum/count



#Can be used to create a utility matrix for the entire dataset , but takes a lot of time to execute
# Need to optimise this approcah further for later

def computeSimilarities(dataframe,userId):
    utility_dict = dict()
    utility_matrix = [ [ 0 for i in range(len(dataframe))] for j in range(len(dataframe.columns))]
    for ir in range(len(dataframe)):
        utility_dict[ir] = list()
        nonzero_initial = np.array(dataframe.iloc[ir].nonzero())
        #print(nonzero_initial.shape)
        for nr in range(len(dataframe)):
            nonzero_nextrow = np.array(dataframe.iloc[nr].nonzero())
            if any(x in nonzero_initial for x in nonzero_nextrow):
                list_common = set(nonzero_initial.flatten()).intersection(set(nonzero_nextrow.flatten()))
                if len(list_common) >= 100:
                    if ir == nr:
                        next
                    else:
                        utility_dict[ir].append(nr)
                    temp = pearsonr(dataframe.iloc[ir,list(list_common)],dataframe.iloc[nr,list(list_common)])
                else:
                    utility_matrix[ir][nr] = 0
                    next
            else:
                utility_matrix[ir][nr] = 0
                next

    return utility_dict,utility_matrix


def computeSimilaritiesforUser(dataframe,userId):
    utility_dict = dict()
    utility_matrix = np.zeros(len(dataframe.columns))
    utility_dict[userId] = list()
    idx = dataframe.index.get_loc(userId)

    nonzero_initial = np.array(dataframe.iloc[idx].nonzero())
    for nr in range(len(dataframe)):
            nonzero_nextrow = np.array(dataframe.iloc[nr].nonzero())
            list_common = set(nonzero_initial.flatten()).intersection(set(nonzero_nextrow.flatten()))
            ## Selecting users that have atleast min 100 movies ratings in common
            if len(list_common) >= 100:
                if idx == nr:
                    utility_matrix[nr] = 0.0
                else:
                    utility_dict[userId].append(nr)
                    temp = pearsonr(dataframe.iloc[idx,list(list_common)],dataframe.iloc[nr,list(list_common)])
                    utility_matrix[nr] = temp[0]
            else:
                utility_matrix[nr] = 0

    return utility_dict,utility_matrix


pivottb = pd.DataFrame(pd.pivot_table(Ratings,index='userId',values='rating',columns='movieId', fill_value=0))


utility_dict,utility_mat = computeSimilaritiesforUser(pivottb,316)

print("Unique Users in the Dataset------->",len((Ratings.userId.unique())))
print("Unique Movies in the Dataset------->",len((Ratings.movieId.unique())))

cor_df = pd.DataFrame(utility_mat,columns=['UserID'])



sim_users = cor_df.sort_values(by=['UserID'],ascending=False)[:10]


def get_user_similar_movies( user1, user2 ):
        common_movies = Ratings[Ratings.userId == user1].merge(Ratings[Ratings.userId == user2],
        on = "movieId",
        how = "inner")

        return common_movies.merge( Movies, on = 'movieId')


def predict_user_similar_movies( user1, user2 ):
        common_movies = Ratings[Ratings.userId == user1].merge(Ratings[Ratings.userId == user2],
        on = "movieId",
        how = "right")

        return common_movies.merge( Movies, on = 'movieId')

for user2 in sim_users.index.values:
    user2 = pivottb.iloc[user2].name
    com_mov = predict_user_similar_movies(316,user2)
    mask = com_mov['rating_x'].isna()

    print("-----------------------------------------------------------------------------")
    print("Recommended Movies for user {0} based on similar user {1}".format(316,user2))

    print(com_mov.loc[mask,['title','rating_y']].sort_values(by='rating_y',ascending=False).head())
    print("-----------------------------------------------------------------------------\n")
