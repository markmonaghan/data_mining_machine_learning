import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
def list_columns():
    movie_header = ['Main_Genre', 'imdb_rating', 'length', 'rank_in_year']
    movies = pd.read_csv('/Users/rohit/Learning Program/group-coursework-ha/data/blockbusters.csv')
    songs = pd.read_csv('/Users/rohit/Learning Program/group-coursework-ha/data/topsongs.csv', encoding='latin-1')
    games = pd.read_csv('/Users/rohit/Learning Program/group-coursework-ha/data/games.csv')
    print(movies.info(memory_usage='deep'))
    print(songs.info(memory_usage='deep'))
    print(games.info(memory_usage='deep'))
    #pd.set_option('display.max_rows', 5)
    #print(games)
    #print(songs)
    print(movies)
    #movies.plot(x="title", y=["imdb_rating","rank_in_year"])
    bins = [0, 2, 4, 5, 6, 7, 9, 10]
    movies['rating_group'] = pd.cut(movies['imdb_rating'], bins)
    movies.groupby("Main_Genre")['year'].value_counts().plot.bar()
    pd.pivot_table(movies, index='rating_group', columns='year',
                   values="Main_Genre", aggfunc='count').plot(kind='bar')
    plt.show()


list_columns()
