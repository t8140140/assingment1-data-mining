import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn as sk

from sklearn.linear_model import LinearRegression


ratings_table = pd.read_csv('../ml-1m/ratings.dat',
                            '::', None, 0)

movies_table = pd.read_csv('../ml-1m/movies.dat',
                           '::', None, 0)

users_table = pd.read_csv('../ml-1m/users.dat',
                          '::', None, 0)


# 1.
mean_rating_ratings = ratings_table['Rating'].mean()

# print(mean_rating_ratings)


# 2.
mean_rating_per_movie = ratings_table.groupby('MovieID')['Rating'].mean()


# print(mean_rating_per_movie)

# .3
mean_rating_per_user = ratings_table.groupby('UserID')['Rating'].mean()


# print(mean_rating_per_user)

# Create a 2-dimensional table Xi,j:Rating , Xi: UserID , Xj: MovieID

# Find model with regards to the user_average


def find_min_arguments(mean, index=['UserID'], columns=['MovieID'], values='Rating', fit_intercept=True):
    users_ratings = pd.pivot_table(
        data=ratings_table, values=values, index=index, columns=columns, fill_value=3)

    reg = LinearRegression(fit_intercept=fit_intercept).fit(
        users_ratings, mean)

    reg_score = reg.score(users_ratings, mean)

    print('Score: ', reg_score)

    print('Coeficient: ', reg.coef_)

    if (fit_intercept):
        print('Intercept :', reg.intercept_)

    prediction = reg.predict(users_ratings)
    print('Prediction :', (prediction > 5).unique())

    return reg


# find_min_arguments(mean_rating_per_movie)
find_min_arguments(mean_rating_per_user)
find_min_arguments(mean_rating_per_movie, index=[
                   'MovieID'], columns=['UserID'], values='Rating')


# coefficient_determination = reg.score(users_ratings, mean_rating_per_user)


# print(coefficient_determination)

# print(reg.predict(users_ratings))
