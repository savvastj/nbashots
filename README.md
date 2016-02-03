nbashots
=======================================

<div class="row">
<img src="https://i.imgur.com/Hg4gg6T.png" height="255" width="261">
<img src="https://i.imgur.com/Fc3ZTTP.png" height="255" width="261">
<img src="https://i.imgur.com/xw0Jlm3.png" height="255" width="261">
</div>

`nbashots` is a library that is built on top of matplotlib, seaborn, and
bokeh in order to create a variety of NBA shot charts using Python.
`nbashots` allows for easy access to the NBA stats API in order to extract
the necessary data for creating shot charts.

Just note that this library is in early development but it should work for Python
2.7 and 3.3+. Most of the code is based on my 
[blog post](http://savvastjortjoglou.com/nba-shot-sharts.html).


Requirements
------------
- Python 2.7 or 3.3+

### Mandatory packages

- [bokeh](http://bokeh.pydata.org/en/latest/)

- [matplotlib](http://matplotlib.sourceforge.net)

- [numpy](http://www.numpy.org/)

- [pandas](http://pandas.pydata.org/)

- [requests](http://docs.python-requests.org/en/latest/)

- [scipy](http://www.scipy.org/)

- [seaborn](https://stanford.edu/~mwaskom/software/seaborn/) == 0.6.0


Installation
------------
To install just run:
    
    pip install nbashots

Tutorial
--------
You can check out a tutorial I wrote up over [here](http://nbviewer.jupyter.org/github/savvastj/nbashots/blob/master/tutorial/Tutorial.ipynb).


TODO
----

- Finish up the documentation and create a readthedocs page.
- Write tests.


License
-------
Released under BSD 3-clause License
