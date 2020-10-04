# Modeling the Epidemic Outbreak and Dynamics of the coronavirus disease (COVID-19)

Initially, this Python package was developed as a part of the graduate course in [Modeling and Simulation of Physical Systems](https://nastava.fesb.unist.hr/nastava/predmeti/11623) and for the paper which can be accessed [here](https://arxiv.org/abs/2005.01434). Currently, this is an ongoing project that provides the ability to model the initial growth of the infected individuals and the dynamics of the epidemic, fitting the corresponding curves, observing the epidemic situation through descriptive statistics and calculating epidemiological parameters e.g. expected disease duration, transmission coefficient, reproductive number etc.

Citation
--------
```citation
@article{kapetanovic2020covid,
    title={{Modeling the Epidemic Outbreak and Dynamics of COVID-19 in Croatia}},
    author={Ante Lojic Kapetanovic and Dragan Poljak},
    year={2020},
    eprint={2005.01434},
    archivePrefix={arXiv},
    primaryClass={q-bio.PE}
}
```

## Requirements 

Check the [`conda_env.yml`](https://github.com/antelk/covid-19/blob/master/conda_env.yml) file.

```bash
$ conda update conda
$ conda install git
$ git clone https://github.com/antelk/covid-19
$ cd covid-19
$ conda env create -f conda_env.yml
```

## Installation

Install from the source using pip

```bash
$ git clone https://github.com/antelk/covid-19 # if it is not already cloned
$ cd covid-19
$ pip install .
```

or directly using the standard python `setup.py` installation

```bash
$ python setup.py install
```

## Running a simulation

See [Examples](#Examples) section.

## Examples

Go to the [sandbox](https://github.com/antelk/covid-19/tree/master/sandbox) directory and fire up jupyter notebook.
There are currently 6 notebooks, all exploring the COVID-19 situation in Croatia. To view it online click on following links:
* [Data visualization and trends](https://github.com/antelk/covid-19/blob/master/sandbox/00-Data-Visualization-and-Trends.ipynb)
* [Exploring mortality](https://github.com/antelk/covid-19/blob/master/sandbox/01-Exploring-Mortality.ipynb)
* [Outbreak exponential fitting](https://github.com/antelk/covid-19/blob/master/sandbox/02-Epidemic-Growth-Modeling.ipynb)
* [Modified SEIR simulation](https://github.com/antelk/covid-19/blob/master/sandbox/03-Modified-SEIR-Simulation.ipynb)
* [SEIRD simulation with confidence intervals based on RT-PCR test sensitivity](https://github.com/antelk/covid-19/blob/master/sandbox/04-SEIRD-Simulation.ipynb)
* [Multiple outbreaks simulation](https://github.com/antelk/covid-19/blob/master/sandbox/05-Multiple-Waves-Simulation.ipynb)
* [Reproduction number dynamics in time](https://github.com/antelk/covid-19/blob/master/sandbox/06-Basic-Reproduction-Number-Time-Series.ipynb)

## License

[MIT](https://github.com/antelk/covid-19/blob/master/LICENSE)