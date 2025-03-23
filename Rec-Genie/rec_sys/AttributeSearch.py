


"""
This module is used to search for attributes in the dataset. E.g. query the datasets for a film name, actor, director, etc.

There will be X functions in this module:
- search_by_film_name
- search_by_actor_name
- search_by_director_name
- search_by_genre
- search_by_year

Each function will take in a string as input.
"""
import pandas as pd


class AttributeSearch:
    def __init__(self, films_path, credits_path):
        self.films_df = pd.read_csv(films_path)
        self.credits_df = pd.read_csv(credits_path)

    def search_by_film_name(self, film_name: str):
        """
        Search for a film by its name.

        Args:
            film_name (str): The name of the film to search for.

        Returns:
            pandas.DataFrame: DataFrame containing the film(s) that match the search query.
        """

        return  self.films_df[self.films_df['title'].str.contains(film_name, case=False)]

        pass

    def search_films_by_actor_name(self, actor_name: str):
        """
        Search for a film by an actor's name.

        Args:
            actor_name (str): The name of the actor to search for.

        Returns:
            pandas.DataFrame: DataFrame containing the film(s) that match the search query.
        """

        return self.films_df[self.films_df['cast'].str.contains(actor_name, case=False)]
        pass


    def search_films_by_director_name(self, director_name: str):
        """
        Search for a film by a director's name.

        Args:
            director_name (str): The name of the director to search for.

        Returns:
            pandas.DataFrame: DataFrame containing the film(s) that match the search query.
        """
        pass

    def search_actor_id_by_name(self, actor_name: str):
        """
        Search for an actor's ID by their name.

        Args:
            actor_name (str): The name of the actor to search for.

        Returns:
            int: The ID of the actor.
        """
        pass

    def search_director_id_by_name(self, director_name: str):
        """
        Search for a director's ID by their name.

        Args:
            director_name (str): The name of the director to search for.

        Returns:
            int: The ID of the director.
        """
        pass

    def search_genre_id_by_name(self, genre_name: str):
        """
        Search for a genre's ID by its name.

        Args:
            genre_name (str): The name of the genre to search for.

        Returns:
            int: The ID of the genre.
        """
        pass



