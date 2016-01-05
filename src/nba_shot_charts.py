import requests
import numpy as np
from scipy.stats import binned_statistic_2d
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import seaborn as sns
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool
from math import pi

try:
    from urllib import urlretrieve  # python 2 compatible
except:
    from urllib.request import urlretrieve  # python 3 compatible

sns.set_style('white')
sns.set_color_codes()


class Shots(object):
    """
    Shots is a wrapper around the NBA stats API that can access the shot chart
    data.
    """
    def __init__(self, player_id, league_id="00", season="2015-16",
                 season_type="Regular Season", team_id=0, game_id="",
                 outcome="", location="", month=0, season_segment="",
                 date_from="", date_to="", opp_team_id=0, vs_conference="",
                 vs_division="", position="", rookie_year="", game_segment="",
                 period=0, last_n_games=0, clutch_time="", ahead_behind="",
                 point_diff="", range_type="", start_period="", end_period="",
                 start_range="", end_range="", context_filter="",
                 context_measure="FGA"):

        self.player_id = player_id

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
                                "Position": position,
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

        self.response = requests.get(self.base_url, params=self.url_paramaters)

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
    response = requests.get(url)
    # access 'resultSets', which is a list containing the dict with all the data
    # The 'header' key accesses the headers
    headers = response.json()['resultSets'][0]['headers']
    # The 'rowSet' key contains the player data along with their IDs
    players = response.json()['resultSets'][0]['rowSet']
    # Create dataframe with proper numeric types
    df = pd.DataFrame(players, columns=headers)
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
    """Returns a pandas DataFrame with all team IDs"""
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
    Returns the image of the player from stats.nba.com and saves it in
    the current directory.

    Parameters
    ----------
    player_id: int
        The player ID used to find the image.

    """
    url = "http://stats.nba.com/media/players/230x185/"+str(player_id)+".png"
    img_file = str(player_id) + ".png"
    pic = urlretrieve(url, img_file)
    return pic[0]


def draw_court(ax=None, color='gray', lw=1, outer_lines=False):
    """Returns an axes with a basketball court drawn onto to it.

    This function draws a court based on the x and y-axis values that the NBA
    stats API provides for the shot chart data.  For example the center of the
    hoop is located at the (0,0) coordinate.  Twenty-two feet from the left of
    the center of the hoop in is represented by the (-220,0) coordinates.
    So one foot equals +/-10 units on the x and y-axis.

    Parameters
    ----------
    ax : Axes, optional
        The Axes object to plot the court onto.
    color : matplotlib color, optional
        The color of the court lines.
    lw : float, optional
        The linewidth the of the court lines.
    outer_lines : boolean, optional
        If `True` it draws the out of bound lines in same style as the rest of
        the court.

    Returns
    -------
    ax : Axes
        The Axes object with the court on it.

    """
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -12.5), 60, 0, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the right side 3pt lines, it's 14ft long before it arcs
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    # Create the right side 3pt lines, it's 14ft long before it arcs
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax


def shot_chart(x, y, kind="scatter", title="", color="b", cmap=None,
               xlim=(-250, 250), ylim=(422.5, -47.5),
               court_color="gray", court_lw=1, outer_lines=True,
               flip_court=False, kde_shade=True, hex_gridsize=None,
               ax=None, despine=True, **kwargs):
    """
    Returns an Axes object with player shots plotted.

    Parameters
    ----------

    x, y : strings or vector
        The x and y coordinates of the shots taken. They can be passed in as
        vectors (such as a pandas Series) or as columns from the pandas
        DataFrame passed into ``data``.
    data : DataFrame, optional
        DataFrame containing shots where ``x`` and ``y`` represent the
        shot location coordinates.
    kind : { "scatter", "kde", "hex" }, optional
        The kind of shot chart to create.
    title : str, optional
        The title for the plot.
    color : matplotlib color, optional
        Color used to plot the shots
    cmap : matplotlib Colormap object or name, optional
        Colormap for the range of data values. If one isn't provided, the
        colormap is derived from the valuue passed to ``color``.
    {x, y}lim : two-tuples, optional
        The axis limits of the plot.  The defaults represent the out of bounds
        lines and half court line.
    court_color : matplotlib color, optional
        The color of the court lines.
    court_lw : float, optional
        The linewidth the of the court lines.
    outer_lines : boolean, optional
        If ``True`` the out of bound lines are drawn in same style as the rest
        of the court.
    flip_court : boolean, optional
        If ``True`` orients the hoop towards the bottom of the plot.  Default
        is ``False``, which orients the court where the hoop is towards the top
        of the plot.
    kde_shade : boolean, optional
        Default is ``True``, which shades in the KDE contours.
    hex_gridsize : int, optional
        Number of hexagons in the x-direction.  The default is calculated using
        the Freedman-Diaconis method.
    ax : Axes, optional
        The Axes object to plot the court onto.
    despine : boolean, optional
        If ``True``, removes the spines.
    kwargs : key, value pairs
        Keyword arguments for matplotlib Collection properties.


    Returns
    -------
     ax : Axes
        The Axes object with the shot chart plotted on it.

    """

    if ax is None:
        ax = plt.gca()

    if cmap is None:
        cmap = sns.light_palette(color, as_cmap=True)

    if not flip_court:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.set_xlim(xlim[::-1])
        ax.set_ylim(ylim[::-1])

    ax.tick_params(labelbottom="off", labelleft="off")
    ax.set_title(title, fontsize=18)

    draw_court(ax, color=court_color, lw=court_lw, outer_lines=outer_lines)

    if kind == "scatter":
        ax.scatter(x, y, c=color, **kwargs)

    elif kind == "kde":
        sns.kdeplot(x, y, shade=kde_shade, cmap=cmap, ax=ax, **kwargs)
        ax.set_xlabel('')
        ax.set_ylabel('')

    elif kind == "hex":
        if hex_gridsize is None:
            # Get the number of bins for hexbin using Freedman-Diaconis rule
            # This is idea was taken from seaborn, which got the calculation
            # from http://stats.stackexchange.com/questions/798/
            from seaborn.distributions import _freedman_diaconis_bins
            x_bin = _freedman_diaconis_bins(x)
            y_bin = _freedman_diaconis_bins(y)
            hex_gridsize = int(np.mean([x_bin, y_bin]))

        ax.hexbin(x, y, gridsize=hex_gridsize, cmap=cmap, **kwargs)

    else:
        raise ValueError("kind must be 'scatter', 'kde', or 'hex'.")

    if despine:
        sns.despine(ax=ax, left=True, bottom=True)

    return ax


def shot_chart_jointgrid(x, y, data=None, joint_type="scatter", title="",
                         joint_color="b", cmap=None,  xlim=(-250, 250),
                         ylim=(422.5, -47.5), court_color="gray", court_lw=1,
                         outer_lines=True, flip_court=False,
                         joint_kde_shade=True, hex_gridsize=None,
                         marginals_color="b", marginals_type="both",
                         marginals_kde_shade=True, size=(12, 11), space=0,
                         despine=True, joint_kws=None, marginal_kws=None,
                         **kwargs):

    """
    Returns a JointGrid object containing the shot chart.

    This function allows for more flexibility in customizing your shot chart
    than the ``shot_chart_jointplot`` function.

    Parameters
    ----------

    x, y : strings or vector
        The x and y coordinates of the shots taken. They can be passed in as
        vectors (such as a pandas Series) or as columns from the pandas
        DataFrame passed into ``data``.
    data : DataFrame, optional
        DataFrame containing shots where ``x`` and ``y`` represent the shot
        location coordinates.
    joint_type : { "scatter", "kde", "hex" }, optional
        The type of shot chart for the joint plot.
    title : str, optional
        The title for the plot.
    joint_color : matplotlib color, optional
        Color used to plot the shots on the joint plot.
    cmap : matplotlib Colormap object or name, optional
        Colormap for the range of data values. If one isn't provided, the
        colormap is derived from the valuue passed to ``color``.
    {x, y}lim : two-tuples, optional
        The axis limits of the plot.  The defaults represent the out of bounds
        lines and half court line.
    court_color : matplotlib color, optional
        The color of the court lines.
    court_lw : float, optional
        The linewidth the of the court lines.
    outer_lines : boolean, optional
        If ``True`` the out of bound lines are drawn in same style as the rest
        of the court.
    flip_court : boolean, optional
        If ``True`` orients the hoop towards the bottom of the plot. Default is
        ``False``, which orients the court where the hoop is towards the top of
        the plot.
    joint_kde_shade : boolean, optional
        Default is ``True``, which shades in the KDE contours on the joint plot.
    hex_gridsize : int, optional
        Number of hexagons in the x-direction.  The default is calculated using
        the Freedman-Diaconis method.
    marginals_color : matplotlib color, optional
        Color used to plot the shots on the marginal plots.
    marginals_type : { "both", "hist", "kde"}, optional
        The type of plot for the marginal plots.
    marginals_kde_shade : boolean, optional
        Default is ``True``, which shades in the KDE contours on the marginal
        plots.
    size : tuple, optional
        The width and height of the plot in inches.
    space : numeric, optional
        The space between the joint and marginal plots.
    despine : boolean, optional
        If ``True``, removes the spines.
    {joint, marginal}_kws : dicts
        Additional kewyord arguments for joint and marginal plot components.
    kwargs : key, value pairs
        Keyword arguments for matplotlib Collection properties.

    Returns
    -------
     grid : JointGrid
        The JointGrid object with the shot chart plotted on it.

    """

    # The joint_kws and marginal_kws idea was taken from seaborn
    # Create the default empty kwargs for joint and marginal plots
    if joint_kws is None:
        joint_kws = {}
    joint_kws.update(kwargs)

    if marginal_kws is None:
        marginal_kws = {}

    # If a colormap is not provided, then it is based off of the joint_color
    if cmap is None:
        cmap = sns.light_palette(joint_color, as_cmap=True)

    # Flip the court so that the hoop is by the bottom of the plot
    if flip_court:
        xlim = xlim[::-1]
        ylim = ylim[::-1]

    # Create the JointGrid to draw the shot chart plots onto
    grid = sns.JointGrid(x=x, y=y, data=data, xlim=xlim, ylim=ylim,
                         space=space)

    # Joint Plot
    # Create the main plot of the joint shot chart
    if joint_type == "scatter":
        grid = grid.plot_joint(plt.scatter, color=joint_color, **joint_kws)

    elif joint_type == "kde":
        grid = grid.plot_joint(sns.kdeplot, cmap=cmap,
                               shade=joint_kde_shade, **joint_kws)

    elif joint_type == "hex":
        if hex_gridsize is None:
            # Get the number of bins for hexbin using Freedman-Diaconis rule
            # This is idea was taken from seaborn, which got the calculation
            # from http://stats.stackexchange.com/questions/798/
            from seaborn.distributions import _freedman_diaconis_bins
            x_bin = _freedman_diaconis_bins(x)
            y_bin = _freedman_diaconis_bins(y)
            hex_gridsize = int(np.mean([x_bin, y_bin]))

        grid = grid.plot_joint(plt.hexbin, gridsize=hex_gridsize, cmap=cmap,
                               **joint_kws)

    else:
        raise ValueError("joint_type must be 'scatter', 'kde', or 'hex'.")

    # Marginal plots
    # Create the plots on the axis of the main plot of the joint shot chart.
    if marginals_type == "both":
        grid = grid.plot_marginals(sns.distplot, color=marginals_color,
                                   **marginal_kws)

    elif marginals_type == "hist":
        grid = grid.plot_marginals(sns.distplot, color=marginals_color,
                                   kde=False, **marginal_kws)

    elif marginals_type == "kde":
        grid = grid.plot_marginals(sns.kdeplot, color=marginals_color,
                                   shade=marginals_kde_shade, **marginal_kws)

    else:
        raise ValueError("marginals_type must be 'both', 'hist', or 'kde'.")

    # Set the size of the joint shot chart
    grid.fig.set_size_inches(size)

    # Extract the the first axes, which is the main plot of the
    # joint shot chart, and draw the court onto it
    ax = grid.fig.get_axes()[0]
    draw_court(ax, color=court_color, lw=court_lw, outer_lines=outer_lines)

    # Get rid of the axis labels
    grid.set_axis_labels(xlabel="", ylabel="")
    # Get rid of all tick labels
    ax.tick_params(labelbottom="off", labelleft="off")
    # Set the title above the top marginal plot
    ax.set_title(title, y=1.2, fontsize=18)

    if despine:
        sns.despine(ax=ax, left=True, bottom=True)

    return grid


def shot_chart_jointplot(x, y, data=None, kind="scatter", title="", color="b",
                         cmap=None, xlim=(-250, 250), ylim=(422.5, -47.5),
                         court_color="gray", court_lw=1, outer_lines=False,
                         flip_court=False, size=(12, 11), space=0,
                         despine=True, joint_kws=None, marginal_kws=None,
                         **kwargs):
    """
    Returns a seaborn JointGrid using sns.jointplot

    Parameters
    ----------

    x, y : strings or vector
        The x and y coordinates of the shots taken. They can be passed in as
        vectors (such as a pandas Series) or as columns from the pandas
        DataFrame passed into ``data``.
    data : DataFrame, optional
        DataFrame containing shots where ``x`` and ``y`` represent the
        shot location coordinates.
    kind : { "scatter", "kde", "hex" }, optional
        The kind of shot chart to create.
    title : str, optional
        The title for the plot.
    color : matplotlib color, optional
        Color used to plot the shots
    cmap : matplotlib Colormap object or name, optional
        Colormap for the range of data values. If one isn't provided, the
        colormap is derived from the valuue passed to ``color``.
    {x, y}lim : two-tuples, optional
        The axis limits of the plot.  The defaults represent the out of bounds
        lines and half court line.
    court_color : matplotlib color, optional
        The color of the court lines.
    court_lw : float, optional
        The linewidth the of the court lines.
    outer_lines : boolean, optional
        If ``True`` the out of bound lines are drawn in same style as the rest
        of the court.
    flip_court : boolean, optional
        If ``True`` orients the hoop towards the bottom of the plot.  Default
        is ``False``, which orients the court where the hoop is towards the top
        of the plot.
    kde_shade : boolean, optional
        Default is ``True``, which shades in the KDE contours.
    hex_gridsize : int, optional
        Number of hexagons in the x-direction.  The default is calculated using
        the Freedman-Diaconis method.
    size : tuple, optional
        The width and height of the plot in inches.
    space : numeric, optional
        The space between the joint and marginal plots.
    {joint, marginal}_kws : dicts
        Additional kewyord arguments for joint and marginal plot components.
    kwargs : key, value pairs
        Keyword arguments for matplotlib Collection properties.

    Returns
    -------
     grid : JointGrid
        The JointGrid object with the shot chart plotted on it.

   """

    # If a colormap is not provided, then it is based off of the color
    if cmap is None:
        cmap = sns.light_palette(color, as_cmap=True)

    if kind not in ["scatter", "kde", "hex"]:
        raise ValueError("kind must be 'scatter', 'kde', or 'hex'.")

    grid = sns.jointplot(x, y, data=None, stat_func=None, kind=kind, space=0,
                         color=color, cmap=cmap, joint_kws=None,
                         marginal_kws=None, **kwargs)

    grid.fig.set_size_inches(size)

    # A joint plot has 3 Axes, the first one called ax_joint
    # is the one we want to draw our court onto and adjust some other settings
    ax = grid.ax_joint

    if not flip_court:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.set_xlim(xlim[::-1])
        ax.set_ylim(ylim[::-1])

    draw_court(ax, color=court_color, lw=court_lw, outer_lines=outer_lines)

    # Get rid of axis labels and tick marks
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelbottom='off', labelleft='off')

    # Add a title
    ax.set_title(title, y=1.2, fontsize=18)

    if despine:
        sns.despine(ax=ax, left=True, bottom=True)

    return grid


def heatmap_fgp(x, y, z, bins=20, title="", cmap=plt.cm.YlOrRd,
                xlim=(-250, 250), ylim=(422.5, -47.5),
                facecolor='lightgray', facecolor_alpha=0.4,
                court_color="black", court_lw=0.5, outer_lines=False,
                flip_court=False, ax=None, **kwargs):

    """
    Returns an AxesImage object that contains a heatmap of the FG%

    TODO: Explain parameters
    """

    # Bin the FGA (x, y) and Calculcate the mean number of times shot was
    # made (z) within each bin
    # mean is the calculated FG percentage for each bin
    mean, xedges, yedges, binnumber = binned_statistic_2d(x=x, y=y,
                                                          values=z,
                                                          statistic='mean',
                                                          bins=bins)

    if ax is None:
        ax = plt.gca()

    if not flip_court:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.set_xlim(xlim[::-1])
        ax.set_ylim(ylim[::-1])

    ax.tick_params(labelbottom="off", labelleft="off")
    ax.set_title(title, fontsize=18)

    ax.patch.set_facecolor(facecolor)
    ax.patch.set_alpha(facecolor_alpha)

    draw_court(ax, color=court_color, lw=court_lw, outer_lines=outer_lines)

    heatmap = ax.imshow(mean.T, origin='lower', extent=[xedges[0], xedges[-1],
                        yedges[0], yedges[-1]], interpolation='nearest',
                        cmap=plt.cm.YlOrRd)

    return heatmap


# Bokeh Shot Chart
def bokeh_draw_court(figure, line_color='gray', line_width=1):
    """Returns a figure with the basketball court lines drawn onto it

    This function draws a court based on the x and y-axis values that the NBA
    stats API provides for the shot chart data.  For example the center of the
    hoop is located at the (0,0) coordinate.  Twenty-two feet from the left of
    the center of the hoop in is represented by the (-220,0) coordinates.
    So one foot equals +/-10 units on the x and y-axis.

    Parameters
    ----------
    figure : Bokeh figure object
        The Axes object to plot the court onto.
    line_color : str, optional
        The color of the court lines. Can be a a Hex value.
    line_width : float, optional
        The linewidth the of the court lines in pixels.


    Returns
    -------
    figure : Figure
        The Figure object with the court on it.

    """

    # hoop
    figure.circle(x=0, y=0, radius=7.5, fill_alpha=0,
                  line_color=line_color, line_width=line_width)

    # backboard
    figure.line(x=range(-30, 31), y=-12.5, line_color=line_color)

    # The paint
    # outerbox
    figure.rect(x=0, y=47.5, width=160, height=190, fill_alpha=0,
                line_color=line_color, line_width=line_width)
    # innerbox
    # left inner box line
    figure.line(x=-60, y=np.arange(-47.5, 143.5), line_color=line_color,
                line_width=line_width)
    # right inner box line
    figure.line(x=60, y=np.arange(-47.5, 143.5), line_color=line_color,
                line_width=line_width)

    # Restricted Zone
    figure.arc(x=0, y=0, radius=40, start_angle=pi, end_angle=0,
               line_color=line_color, line_width=line_width)

    # top free throw arc
    figure.arc(x=0, y=142.5, radius=60, start_angle=pi, end_angle=0,
               line_color=line_color)

    # bottome free throw arc
    figure.arc(x=0, y=142.5, radius=60, start_angle=0, end_angle=pi,
               line_color=line_color, line_dash="dashed")

    # Three point line
    # corner three point lines
    figure.line(x=-220, y=np.arange(-47.5, 92.5), line_color=line_color,
                line_width=line_width)
    figure.line(x=220, y=np.arange(-47.5, 92.5), line_color=line_color,
                line_width=line_width)
    # # three point arc
    figure.arc(x=0, y=0, radius=237.5, start_angle=3.528, end_angle=-0.3863,
               line_color=line_color, line_width=line_width)

    # add center court
    # outer center arc
    figure.arc(x=0, y=422.5, radius=60, start_angle=0, end_angle=pi,
               line_color=line_color, line_width=line_width)
    # inner center arct
    figure.arc(x=0, y=422.5, radius=20, start_angle=0, end_angle=pi,
               line_color=line_color, line_width=line_width)

    # outer lines, consistting of half court lines and out of bounds lines
    figure.rect(x=0, y=187.5, width=500, height=470, fill_alpha=0,
                line_color=line_color, line_width=line_width)

    return figure


def bokeh_shot_chart(data, x="LOC_X", y="LOC_Y", fill_color="#1f77b4",
                     fill_alpha=0.3, line_alpha=0.3, court_line_color='gray',
                     court_line_width=1, hover_tool=False, tooltips=None):

    # TODO: Settings for hover tooltip
    """
    Returns a figure with both FGA and basketball court lines drawn onto it.

    This function expects data to be a ColumnDataSource with the x and y values
    named "LOC_X" and "LOC_Y".  Otherwise specify x and y.

    Parameters
    ----------

    data : DataFrame
        The DataFrame that contains the shot chart data.
    x, y : str, optional
        The x and y coordinates of the shots taken.
    fill_color : str, optional
        The fill color of the shots. Can be a a Hex value.
    fill_alpha : float, optional
        Alpha value for the shots. Must be a floating point value between 0
        (transparent) to 1 (opaque).
    line_alpha : float, optiona
        Alpha value for the outer lines of the plotted shots. Must be a
        floating point value between 0 (transparent) to 1 (opaque).
    court_line_color : str, optional
        The color of the court lines. Can be a a Hex value.
    court_line_width : float, optional
        The linewidth the of the court lines in pixels.
    hover_tool : boolean, optional
        If ``True``, creates hover tooltip for the plot.
    tooltips : List of tuples, optional
        Provides the information for the the hover tooltip.

    Returns
    -------
    fig : Figure
        The Figure object with the shot chart plotted on it.

    """
    source = ColumnDataSource(data)

    fig = figure(width=700, height=658, x_range=[-250, 250],
                 y_range=[422.5, -47.5], min_border=0,
                 x_axis_type=None, y_axis_type=None,
                 outline_line_color="black")

    fig.scatter(x, y, source=source, size=10, fill_alpha=0.3,
                line_alpha=0.3)

    bokeh_draw_court(fig, line_color='gray')

    if hover_tool:
        hover = HoverTool(renderers=[fig.renderers[0]], tooltips=tooltips)
        fig.add_tools(hover)

    return fig
