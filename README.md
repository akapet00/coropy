# CoroPy

Initially, this Python package was developed as a part of the graduate course in [Modeling and Simulation of Physical Systems](https://nastava.fesb.unist.hr/nastava/predmeti/11623) and for the paper which can be accessed [here](https://ieeexplore.ieee.org/document/9243757). For citation use the following:

A. L. Kapetanović and D. Poljak, "Modeling the Epidemic Outbreak and Dynamics of COVID-19 in Croatia," 2020 5th International Conference on Smart and Sustainable Technologies (SpliTech), Split, Croatia, 2020, pp. 1-5, doi: 10.23919/SpliTech49282.2020.9243757.

or for in bibtex format:

```citation
@inProceedings{kapetanovic2020,
    author={A. L. {Kapetanović} and D. {Poljak}},
    booktitle={2020 5th International Conference on Smart and Sustainable Technologies (SpliTech)},
    title={Modeling the Epidemic Outbreak and Dynamics of COVID-19 in Croatia},
    year={2020},
    pages={1-5},
    doi={10.23919/SpliTech49282.2020.9243757}}
```

Currently, this is an ongoing project that provides the ability to model the initial growth of the infected individuals and the dynamics of the epidemic, fitting the corresponding curves, observing the epidemic situation through descriptive statistics and calculating epidemiological parameters e.g. expected disease duration, transmission coefficient, reproductive number, etc.



## Requirements 

Check the [`conda_env.yml`](https://github.com/antelk/coropy/blob/master/conda_env.yml) file.

```bash
$ conda update conda
$ conda install git
$ git clone https://github.com/antelk/coropy
$ cd coropy
$ conda env create -f conda_env.yml
```

## Installation

Install from the source using pip

```bash
$ git clone https://github.com/antelk/coropy  # if it is not already cloned
$ cd coropy
$ pip install .
```

or directly using the standard python `setup.py` installation

```bash
$ python setup.py install
```

## Running a simulation

See [Examples](#Examples) section.

## Examples

Go to the [sandbox](https://github.com/antelk/coropy/tree/master/sandbox) directory and fire up jupyter notebook.
There are currently 6 notebooks, all exploring the COVID-19 situation in Croatia. To view it online click on following links:
* [Data visualization and trends](https://github.com/antelk/coropy/blob/master/sandbox/00-Data-Visualization-and-Trends.ipynb)
* [Exploring mortality](https://github.com/antelk/coropy/blob/master/sandbox/01-Exploring-Mortality.ipynb)
* [Outbreak exponential fitting](https://github.com/antelk/coropy/blob/master/sandbox/02-Epidemic-Growth-Modeling.ipynb)
* [Modified SEIR simulation](https://github.com/antelk/coropy/blob/master/sandbox/03-Modified-SEIR-Simulation.ipynb)
* [SEIRD simulation with confidence intervals based on RT-PCR test sensitivity](https://github.com/antelk/coropy/blob/master/sandbox/04-SEIRD-Simulation.ipynb)
* [Multiple outbreaks simulation](https://github.com/antelk/coropy/blob/master/sandbox/05-Multiple-Waves-Simulation.ipynb)
* [Reproduction number dynamics in time](https://github.com/antelk/coropy/blob/master/sandbox/06-Reproduction-Number-Time-Series.ipynb)

## License

[MIT](https://github.com/antelk/coropy/blob/master/LICENSE)