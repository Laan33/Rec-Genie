import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def weighted_rating(v, m, R, C):
    '''
    Calculate the weighted rating

    Args:
    v -> average rating for each item (float)
    m -> minimum votes required to be classified as popular (float)
    R -> average rating for the item (pd.Series)
    C -> average rating for the whole dataset (pd.Series)

    Returns:
    pd.Series
    '''
    return ((v / (v + m)) * R) + ((m / (v + m)) * C)


def assign_popular_based_score(rating_df, item_df, user_col, item_col, rating_col):
    '''
    Assigned popular based score based on the IMDB weighted average.

    Args:
    rating -> pd.DataFrame contains ['item_id', 'rating'] for each user.

    Returns
    popular_items -> pd.DataFrame contains item and IMDB weighted score.
    '''

    # pre processing
    vote_count = (
        rating_df
        .groupby(item_col, as_index=False)
        .agg({user_col: 'count', rating_col: 'mean'})
    )
    vote_count.columns = [item_col, 'vote_count', 'avg_rating']

    # calcuate input parameters
    C = np.mean(vote_count['avg_rating'])
    m = np.percentile(vote_count['vote_count'], 70)
    vote_count = vote_count[vote_count['vote_count'] >= m]
    R = vote_count['avg_rating']
    v = vote_count['vote_count']
    vote_count['weighted_rating'] = weighted_rating(v, m, R, C)

    # post processing
    vote_count = vote_count.merge(item_df, on=[item_col], how='left')
    popular_items = vote_count.loc[:, [item_col, 'genres', 'vote_count', 'avg_rating', 'weighted_rating']]

    return popular_items


# init constant
USER_COL = 'user_id'
ITEM_COL = 'item_id'
RATING_COL = 'rating'
ratings = pd.read_csv('data/ratings.csv')
items = pd.read_csv('data/movies.csv')


# calcualte popularity based
pop_items = assign_popular_based_score(ratings, items, USER_COL, ITEM_COL, RATING_COL)
pop_items = pop_items.sort_values('weighted_rating', ascending=False)

# plot the popularity based on the weighted score
fix, ax = plt.subplots(figsize=(9, 6))
sns.barplot(data=pop_items.head(10),
            y='item_id',
            x='weighted_rating',
            palette='mako');
sns.despine();



