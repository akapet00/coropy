# Modeling the Epidemic Outbreak and Dynamics of COVID-19 in Croatia

This repository contains the code base for the graduate course in [Modeling and Simulation of Physical Systems](https://nastava.fesb.unist.hr/nastava/predmeti/11623) and for the paper which can be accessed [here](https://arxiv.org/abs/2005.01434). 

Abstract
--------
The paper deals with a modeling of the ongoing epidemic caused by Coronavirus disease 2019 (COVID-19) on the closed territory of the Republic of Croatia. Using the official public information on the number of confirmed infected, recovered and deceased individuals, the modified SEIR compartmental model is developed to describe the underlying dynamics of the epidemic. Fitted modified SEIR model provides the prediction of the disease progression in the near future, considering strict control interventions by means of social distancing and quarantine for infected and at-risk individuals introduced at the beginning of COVID-19 spread on February, 25th by Croatian Ministry of Health. Assuming the accuracy of provided data and satisfactory representativeness of the model used, the basic reproduction number is derived. Obtained results portray potential positive developments and justify the stringent precautionary measures introduced by the Ministry of Health. 

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

## Running a simulation

See [Examples](#Examples) section.

## Examples

Go to the [sandbox](https://github.com/antelk/covid-19/tree/master/sandbox) directory and fire up jupyter notebook.
There are currently 6 notebooks, all exploring the COVID-19 situation in Croatia. To view it online click on following links:
* [Data visualization and trends](https://github.com/antelk/covid-19/blob/master/sandbox/00-Data-Visualization-and-Trends.ipynb)
* [Exploring mortality](https://github.com/antelk/covid-19/blob/master/sandbox/01-Exploring-Mortality.ipynb)
* [Outbreak exponential fitting](https://github.com/antelk/covid-19/blob/master/sandbox/02-Second-Wave-Curve-Fitting.ipynb)
* [Modified SEIR simulation](https://github.com/antelk/covid-19/blob/master/sandbox/03-Modified-SEIR-Simulation.ipynb)
* [SEIRD simulation with confidence intervals based on RT-PCR test sensitivity](https://github.com/antelk/covid-19/blob/master/sandbox/04-SEIRD-Simulation.ipynb)
* [Multiple outbreaks simulation](https://github.com/antelk/covid-19/blob/master/sandbox/05-Multiple-Waves-Simulation.ipynb)
* [Reproduction number dynamics in time](https://github.com/antelk/covid-19/blob/master/sandbox/06-Basic-Reproduction-Number-Time-Series.ipynb)

## License

[MIT](https://github.com/antelk/covid-19/blob/master/LICENSE)

