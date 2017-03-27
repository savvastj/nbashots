import requests
import pandas as pd
import matplotlib.pyplot as plt

try:
    from urllib import urlretrieve  # python 2 compatible
except:
    from urllib.request import urlretrieve  # python 3 compatible


# Fix non-browser request issue
HEADERS = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}


class TeamLog(object):
    """
    TeamLog is a wrapper around the NBA stats API that can access the team game
    log data.
    """
    def __init__(self, team_id, league_id="00", season="2015-16",
                 season_type="Regular Season"):

        self.base_url = "http://stats.nba.com/stats/teamgamelog?"

        self.url_paramaters = {
                                "TeamID": team_id,
                                "LeagueID": league_id,
                                "Season": season,
                                "SeasonType": season_type
                            }

        self.response = requests.get(self.base_url, params=self.url_paramaters,
                                     headers=HEADERS)
        self.response.raise_for_status()

    def get_game_logs(self):
        """Returns team game logs as a pandas DataFrame"""
        logs = self.response.json()['resultSets'][0]['rowSet']
        headers = self.response.json()['resultSets'][0]['headers']
        df = pd.DataFrame(logs, columns=headers)
        df.GAME_DATE = pd.to_datetime(df.GAME_DATE)
        return df

    def get_game_id(self, date):
        """Returns the Game ID associated with the date that is passed in.

        Parameters
        ----------
        date : str
            The date associated with the game whose Game ID. The date that is
            passed in can take on a numeric format of MM/DD/YY (like "01/06/16"
            or "01/06/2016") or the expanded Month Day, Year format (like
            "Jan 06, 2016" or "January 06, 2016").

        Returns
        -------
        game_id : str
            The desired Game ID.
        """
        df = self.get_game_logs()
        game_id = df[df.GAME_DATE == date].Game_ID.values[0]
        return game_id

    def update_params(self, parameters):
        """Pass in a dictionary to update url parameters for NBA stats API

        Parameters
        ----------
        parameters : dict
            A dict containing key, value pairs that correspond with NBA stats
            API parameters.

        Returns
        -------
        self : TeamLog
            The TeamLog object containing the updated NBA stats API
            parameters.
        """
        self.url_paramaters.update(parameters)
        self.response = requests.get(self.base_url, params=self.url_paramaters,
                                     headers=HEADERS)
        # raise error if status code is not 200
        self.response.raise_for_status()
        return self


class PlayerLog(object):
    """
    PlayerLog is a wrapper around the NBA stats API that can access the player
    game log data.
    """
    def __init__(self, player_id, league_id="00", season="2015-16",
                 season_type="Regular Season"):

        self.base_url = "http://stats.nba.com/stats/playergamelog?"

        self.url_paramaters = {
                                "PlayerID": player_id,
                                "LeagueID": league_id,
                                "Season": season,
                                "SeasonType": season_type
                            }

        self.response = requests.get(self.base_url, params=self.url_paramaters,
                                     headers=HEADERS)

        self.response.raise_for_status()

    def get_game_logs(self):
        """Returns player game logs as a pandas DataFrame"""
        logs = self.response.json()['resultSets'][0]['rowSet']
        headers = self.response.json()['resultSets'][0]['headers']
        df = pd.DataFrame(logs, columns=headers)
        df.GAME_DATE = pd.to_datetime(df.GAME_DATE)
        return df

    def get_game_id(self, date):
        """Returns the Game ID associated with the date that is passed in.

        Parameters
        ----------
        date : str
            The date associated with the game whose Game ID. The date that is
            passed in can take on a numeric format of MM/DD/YY (like "01/06/16"
            or "01/06/2016") or the expanded Month Day, Year format (like
            "Jan 06, 2016" or "January 06, 2016").

        Returns
        -------
        game_id : str
            The desired Game ID.
        """
        # Get the game logs
        df = self.get_game_logs()
        game_id = df[df.GAME_DATE == date].Game_ID.values[0]
        return game_id

    def update_params(self, parameters):
        """Pass in a dictionary to update url parameters for NBA stats API

        Parameters
        ----------
        parameters : dict
            A dict containing key, value pairs that correspond with NBA stats
            API parameters.

        Returns
        -------
        self : PlayerLog
            The PlayerLog object containing the updated NBA stats API
            parameters.
        """
        self.url_paramaters.update(parameters)
        self.response = requests.get(self.base_url, params=self.url_paramaters,
                                     headers=HEADERS)
        self.response.raise_for_status()
        return self


class Shots(object):
    """
    Shots is a wrapper around the NBA stats API that can access the shot chart
    data.

    TODO: Explain NBA stats API parameters.
    """
    def __init__(self, player_id=0, team_id=0, league_id="00", season="2015-16",
                 season_type="Regular Season", game_id="", outcome="",
                 location="", month=0, season_segment="", date_from="",
                 date_to="", opp_team_id=0, vs_conference="", vs_division="",
                 PlayerPosition="", rookie_year="", game_segment="", period=0,
                 last_n_games=0, clutch_time="", ahead_behind="", point_diff="",
                 range_type="", start_period="", end_period="", start_range="",
                 end_range="", context_filter="", context_measure="FGA"):

        self.base_url = "http://stats.nba.com/stats/shotchartdetail?"

        # TODO: Figure out what all these parameters mean for NBA stats api
        #       Need to figure out and include CFID and CFPARAMS, they are
        #       associated w/ContextFilter somehow
        self.url_paramaters = {
                                "LeagueID": league_id,
                                "Season": season,
                                "SeasonType": season_type,
                                "TeamID": team_id,
                                "PlayerID": player_id,
                                "GameID": game_id,
                                "Outcome": outcome,
                                "Location": location,
                                "Month": month,
                                "SeasonSegment": season_segment,
                                "DateFrom": date_from,
                                "DateTo": date_to,
                                "OpponentTeamID": opp_team_id,
                                "VsConference": vs_conference,
                                "VsDivision": vs_division,
                                "PlayerPosition": PlayerPosition,
                                "RookieYear": rookie_year,
                                "GameSegment": game_segment,
                                "Period": period,
                                "LastNGames": last_n_games,
                                "ClutchTime": clutch_time,
                                "AheadBehind": ahead_behind,
                                "PointDiff": point_diff,
                                "RangeType": range_type,
                                "StartPeriod": start_period,
                                "EndPeriod": end_period,
                                "StartRange": start_range,
                                "EndRange": end_range,
                                "ContextFilter": context_filter, # unsure of what this does
                                "ContextMeasure": context_measure
                            }

        self.response = requests.get(self.base_url, params=self.url_paramaters,
                                     headers=HEADERS)
        self.response.raise_for_status()

    def get_shots(self):
        """Returns the shot chart data as a pandas DataFrame."""
        shots = self.response.json()['resultSets'][0]['rowSet']
        headers = self.response.json()['resultSets'][0]['headers']
        return pd.DataFrame(shots, columns=headers)

    def get_league_avg(self):
        """Returns the league average shooting stats for all FGA in each zone"""
        shots = self.response.json()['resultSets'][1]['rowSet']
        headers = self.response.json()['resultSets'][1]['headers']
        return pd.DataFrame(shots, columns=headers)

    def update_params(self, parameters):
        """Pass in a dictionary to update url parameters for NBA stats API

        Parameters
        ----------
        parameters : dict
            A dict containing key, value pairs that correspond with NBA stats
            API parameters.

        Returns
        -------
        self : Shots
            The Shots object containing the updated NBA stats API parameters.
        """
        self.url_paramaters.update(parameters)
        self.response = requests.get(self.base_url, params=self.url_paramaters,
                                     headers=HEADERS)
        self.response.raise_for_status()
        return self


def get_all_player_ids(ids="shots"):
    """
    Returns a pandas DataFrame containing the player IDs used in the
    stats.nba.com API.

    Parameters
    ----------
    ids : { "shots" | "all_players" | "all_data" }, optional
        Passing in "shots" returns a DataFrame that contains the player IDs of
        all players have shot chart data.  It is the default parameter value.

        Passing in "all_players" returns a DataFrame that contains
        all the player IDs used in the stats.nba.com API.

        Passing in "all_data" returns a DataFrame that contains all the data
        accessed from the JSON at the following url:
        http://stats.nba.com/stats/commonallplayers?IsOnlyCurrentSeason=0&LeagueID=00&Season=2015-16

        The column information for this DataFrame is as follows:
            PERSON_ID: The player ID for that player
            DISPLAY_LAST_COMMA_FIRST: The player's name.
            ROSTERSTATUS: 0 means player is not on a roster, 1 means he's on a
                          roster
            FROM_YEAR: The first year the player played.
            TO_YEAR: The last year the player played.
            PLAYERCODE: A code representing the player. Unsure of its use.

    Returns
    -------
    df : pandas DataFrame
        The pandas DataFrame object that contains the player IDs for the
        stats.nba.com API.

    """
    url = "http://stats.nba.com/stats/commonallplayers?IsOnlyCurrentSeason=0&LeagueID=00&Season=2015-16"

    # get the web page
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    # access 'resultSets', which is a list containing the dict with all the data
    # The 'header' key accesses the headers
    headers = response.json()['resultSets'][0]['headers']
    # The 'rowSet' key contains the player data along with their IDs
    players = response.json()['resultSets'][0]['rowSet']
    # Create dataframe with proper numeric types
    df = pd.DataFrame(players, columns=headers)

    # Dealing with different means of converision for pandas 0.17.0 or 0.17.1
    # and 0.15.0 or loweer
    if '0.17' in pd.__version__:
        # alternative to convert_objects() to numeric to get rid of warning
        # as convert_objects() is deprecated in pandas 0.17.0+
        df = df.apply(pd.to_numeric, args=('ignore',))
    else:
        df = df.convert_objects(convert_numeric=True)

    if ids == "shots":
        df = df.query("(FROM_YEAR >= 2001) or (TO_YEAR >= 2001)")
        df = df.reset_index(drop=True)
        # just keep the player ids and names
        df = df.iloc[:, 0:2]
        return df
    if ids == "all_players":
        df = df.iloc[:, 0:2]
        return df
    if ids == "all_data":
        return df
    else:
        er = "Invalid 'ids' value. It must be 'shots', 'all_shots', or 'all_data'."
        raise ValueError(er)


def get_player_id(player):
    """
    Returns the player ID(s) associated with the player name that is passed in.

    There are instances where players have the same name so there are multiple
    player IDs associated with it.

    Parameters
    ----------
    player : str
        The desired player's name in 'Last Name, First Name' format. Passing in
        a single name returns a numpy array containing all the player IDs
        associated with that name.

    Returns
    -------
    player_id : numpy array
        The numpy array that contains the player ID(s).

    """
    players_df = get_all_player_ids("all_data")
    player = players_df[players_df.DISPLAY_LAST_COMMA_FIRST == player]
    # if there are no plyaers by the given name, raise an a error
    if len(player) == 0:
        er = "Invalid player name passed or there is no player with that name."
        raise ValueError(er)
    player_id = player.PERSON_ID.values
    return player_id


def get_all_team_ids():
    """Returns a pandas DataFrame with all Team IDs"""
    df = get_all_player_ids("all_data")
    df = pd.DataFrame({"TEAM_NAME": df.TEAM_NAME.unique(),
                       "TEAM_ID": df.TEAM_ID.unique()})
    return df


def get_team_id(team_name):
    """ Returns the team ID associated with the team name that is passed in.

    Parameters
    ----------
    team_name : str
        The team name whose ID we want.  NOTE: Only pass in the team name
        (e.g. "Lakers"), not the city, or city and team name, or the team
        abbreviation.

    Returns
    -------
    team_id : int
        The team ID associated with the team name.

    """
    df = get_all_team_ids()
    df = df[df.TEAM_NAME == team_name]
    if len(df) == 0:
        er = "Invalid team name or there is no team with that name."
        raise ValueError(er)
    team_id = df.TEAM_ID.iloc[0]
    return team_id


def get_player_img(player_id):
    """
    Returns the image of the player from stats.nba.com as a numpy array and
    saves the image as PNG file in the current directory.

    Parameters
    ----------
    player_id: int
        The player ID used to find the image.

    Returns
    -------
    player_img: ndarray
        The multidimensional numpy array of the player image, which matplotlib
        can plot.
    """
    url = "http://stats.nba.com/media/players/230x185/"+str(player_id)+".png"
    img_file = str(player_id) + ".png"
    pic = urlretrieve(url, img_file)
    player_img = plt.imread(pic[0])
    return player_img
