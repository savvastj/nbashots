# Always prefer setuptools over distutils
from setuptools import setup
# To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

long_description = """\
nba_shot_charts is a library that is built on top of matplotlib, seaborn, and
bokeh in order to create a variety of NBA shot charts using Python.
nba_shot_charts allows for easy access to the NBA stats API in order to exrtact
the necessary data for creating shot charts.
"""

setup(
    name='nba_shot_charts',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1.0a1',

    description='A visualiztion library that helps create NBA player shot charts.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/savvastj/nba_shot_charts',

    # Author details
    author='Savvas Tjortjoglou',
    author_email='savvas.tjortjoglou@gmail.com',

    # Choose your license
    license='BSD (3-clause) "New" or "Revised License"',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering :: Visualization',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'
    ],

    # What does your project relate to?
    keywords='nba data visualiztion shot charts',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    # packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['bokeh', 'matplotlib', 'numpy', 'pandas', 'requests',
                      'scipy', 'seaborn==0.6.0']
)
