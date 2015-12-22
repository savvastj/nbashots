def get_player_id(player):
    """
    Loads a pandas DataFrame, numpy array, or int with the desired player ID(s)
    from an online repository.

    The player IDs are used to identify players in the NBA stats api.

    Parameters
    ----------
    player : str
        The desired player's name in 'Last Name, First Name' format. Passing in
        a single name returns a numpy array containing all the player IDs
        associated with that name.

        Passing "SHOTS" returns a DataFrame with all the players and their IDs
        that have shot chart data.

        Passing in "ALL" returns a DataFrame with all the available player IDs
        used by the NBA stats API, along with additional information.

        The column information for this DataFrame is as follows:
            PERSON_ID: The player ID for that player
            DISPLAY_LAST_COMMA_FIRST: The player's name.
            ROSTERSTATUS: 0 means player is not on a roster, 1 means he's on a
                          roster
            FROM_YEAR: The first year the player played.
            TO_YEAR: The last year the player played.
            PLAYERCODE: A code representing the player. Unsure of its use.
    """

    url = "http://stats.nba.com/stats/commonallplayers?IsOnlyCurrentSeason=0&LeagueID=00&Season=2015-16"

    # get the web page
    response = requests.get(url)
    # access 'resultSets', which is a list containing the dict with all the data.
    # The 'header' key accesses the headers
    headers = response.json()['resultSets'][0]['headers']
    # The 'rowSet' key contains the player data along with their IDs
    players = response.json()['resultSets'][0]['rowSet']
    players_df = pd.DataFrame(players, columns=headers)

    if player == "shots":
        # just keep the player ids and names
        player_ids = players_df.iloc[:,0:2]
        # rename the columns
        player_ids.columns = ["player_id", "player_name"]
        return player_ids
    elif player == "all":
        return players_df
    else:
        player_id = players_df[players_df.DISPLAY_LAST_COMMA_FIRST == player].PERSON_ID
        if len(player_id) == 1:
            return player_id.values[0]
        if len(player_id) == 0:
            raise NoPlayerError('There is no player with that name.')
        return player_id.values
