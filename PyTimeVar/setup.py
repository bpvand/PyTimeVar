import setuptools
import os

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# def read(fname):
#     return open(os.path.join(os.path.dirname(__file__), fname)).read()

setuptools.setup(
    name='PyTimeVar',
    version="0.0.12",
    author='Mingxuan Song, Bernhard van der Sluis, Yicong Lin',
    author_email='678270ms@eur.nl, vandersluis@ese.eur.nl, yc.lin@vu.nl',
    description = ("The PyTimeVar package offers state-of-the-art estimation and statistical inference methods for time series regression models with flexible trends and/or time- varying coefficients."),
    python_requires=">=3.9",
    keywords = 'time-varying, bootstrap, nonparametric estimation, filtering',
    packages=['PyTimeVar', 'PyTimeVar.bhpfilter', 'PyTimeVar.locallinear','PyTimeVar.kalman', 'PyTimeVar.powerlaw', 'PyTimeVar.gas', 'PyTimeVar.datasets', \
              'PyTimeVar.datasets.co2', 'PyTimeVar.datasets.inflation', 'PyTimeVar.datasets.herding','PyTimeVar.datasets.temperature', 'PyTimeVar.datasets.usd'],
    install_requires=['numpy', 'pandas', 'matplotlib',
                      'scipy', 'statsmodels', 'tqdm'],
    # long_description=read('README.md'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = "https://github.com/bpvand/PyTimeVar",
    license = 'GPLv3+',
    classifier=['License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',],
    include_package_data=True,
    package_data={"PyTimeVar.datasets": ["**/*.csv"],}
)
