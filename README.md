# PyTimeVar: A Python Package for Trending Time-Varying Time Series Models
<!-- badges: start -->
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![PyPI](https://img.shields.io/pypi/v/PyTimeVar?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/PyTimeVar)
<!-- badges: end -->

Authors: Mingxuan Song (m3.song@student.vu.nl, Vrije Universiteit Amsterdam), Bernhard van der Sluis (vandersluis@ese.eur.nl, Erasmus Universiteit Rotterdam), and Yicong Lin (yc.lin@vu.nl, Vrije Universiteit Amsterdam & Tinbergen Institute)

## Purpose of the package

The PyTimeVar package offers state-of-the-art estimation and statistical inference methods for time series regression models with flexible trends and/or time-varying coefficients. The package implements nonparametric estimation along with multiple recently proposed bootstrap-assisted inference methods. Pointwise confidence intervals and simultaneous bands of parameter curves via bootstrap can be easily obtained using user-friendly commands. The package also includes four commonly used methods for modeling trends and time-varying relationships: boosted Hodrick-Prescot filter, power-law trend models, state-space models, and score-driven models. This allows users to compare different approaches within a unified environment.

The package is built upon several papers and books. We list the key references below.

### Local linear kernel estimation and bootstrap inference
Friedrich and Lin (2024) (doi: https://doi.org/10.1016/j.jeconom.2022.09.004);
Lin et al. (2024) (doi: https://doi.org/10.1080/10618600.2024.2403705);
Friedrich et al. (2020) (doi: https://doi.org/10.1016/j.jeconom.2019.05.006);
Smeekes and Urbain (2014) (doi: https://doi.org/10.26481/umagsb.2014008)
Zhou and Wu (2010) (doi: https://doi.org/10.1111/j.1467-9868.2010.00743.x);
Bühlmann (1998) (doi: https://doi.org/10.1214/aos/1030563978);


### Boosted HP filter
Mei et al. (2024) (doi: doi: https://doi.org/10.1002/jae.3086);
Biswas et al. (2024) (doi: https://doi.org/10.1080/07474938.2024.2380704);
Phillips and Shi (2021) (doi: https://doi.org/10.1111/iere.12495);


### Power-law trend models
Lin and Reuvers (2024) (https://tinbergen.nl/discussion-paper/6214/22-092-iii-cointegrating-polynomial-regressions-with-power-law-trends-environmental-kuznets-curve-or-omitted-time-effects);
Robinson (2012) (doi: https://doi.org/10.3150/10-BEJ349);


### State-space models
Durbin and Koopman (2012) (doi: https://doi.org/10.1093/acprof:oso/9780199641178.001.0001)

### Score-drive models
Creal et al. (2013) (doi: https://doi.org/10.1002/jae.1279);
Harvey (2013) (doi: https://doi.org/10.1017/CBO9781139540933);

## Features

- Nonparametric estimation of time-varying time series models, along with various bootstrap-assisted methods for inference, including local blockwise wild bootstrap, wild bootstrap, sieve bootstrap, sieve wild bootstrap, autoregressive wild bootstrap
- Alternative estimation methods for modeling trend and time-varying relationships, including boosted HP filter, power-law trend models, state-space, and score-driven models.
- Unified framework for comparison of methods.
- Multiple datasets for illustration.

## Getting started

The PyTimeVar can implemented as a PyPI package. To download the package in your Python environment, use the following command:
```python
pip install PyTimeVar
```

## Support
The documentation of the package can be found at the GitHub repository https://github.com/bpvand/PyTimeVar, and ReadTheDocs https://pytimevar.readthedocs.io/en/latest/.

For any questions or feedback regarding the PyTimeVar package, please feel free to contact the authors via email: 
m3.song@student.vu.nl; 
vandersluis@ese.eur.nl; 
yc.lin@vu.nl.
