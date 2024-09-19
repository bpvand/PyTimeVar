# PyTimeVar: A Python package for Trending Time-Varying Time Series Models
Authors: Mingxuan Song, Bernhard van der Sluis, Yicong Lin

## Purpose of the Package

The PyTimeVar package offers state-of-the-art estimation and statistical inference methods for time series regression models with flexible trends and/or time-
varying coefficients. The package implements nonparametric estimation along with multiple new bootstrap-assisted inference methods.
It provides a range of bootstrap techniques for constructing pointwise confidence intervals and simultaneous bands for parameter curves. 
Additionally, the package includes four widely used methods for modeling trends and time-varying relationships. 
This allows users to compare different approaches within a unified environment.

The package is build upon the methods of several papers and books. We list the key references below.
### Local linear regression and bootstrap inference
Bühlmann (1998) (doi: 10.1214/aos/1030563978); Zhou and Wu (2010) (doi: https://doi.org/10.1111/j.1467-9868.2010.00743.x); Friedrich et al. (2020, doi: https://doi.org/10.1016/j.jeconom.2019.05.006); Friedrich
and Lin (2024, https://doi.org/10.1016/j.jeconom.2022.09.004); Lin et al. (2024) (doi: https://doi.org/10.1080/10618600.2024.2403705)

### Boosted HP filter
Phillips and Shi (2021) (doi: https://doi.org/10.1111/iere.12495);  Mei et al. (2024) (doi: doi:https://doi.org/10.1002/jae)
3086)

### Power-law trend models
Robinson (2012) (doi: 10.3150/10-BEJ349); Lin and Reuvers (2024)

### State-space models
Durbin and Koopman (2012) (doi: 10.1093/acprof:oso/9780199641178.001.0001.)

### Score-drive models 
Harvey (2013) (doi: 
https://doi.org/10.1017/CBO9781139540933); Creal et al. (2013) (doi: https://doi.org/10.
1002/jae.1279.)

## Features

- Nonparametric estimation of time-varying time series models, along with multiple bootstrap-assisted inference methods
- Alternative estimation methods for modelling trend and time-varying relationships.
- Unified framework for comparison of methods.
- Four datasets for illustration.

## Getting Started

The PyTimeVar can implemented as a PyPI package. To download the package in your Python environment, use the following command:
```python 
pip install PyTimeVar
```

## Support
The documentation of the package can found at the github repository https://github.com/bpvand/PyTimeVar, and at ReadTheDocs https://pytimevar.readthedocs.io/en/latest/
Any questions and comments on the PyTimeVar package can be raised via email to vandersluis@ese.eur.nl.
