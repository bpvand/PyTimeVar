import setuptools
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setuptools.setup(
    name='PyTimeVar',
    version="0.0.6",
    author='Mingxuan Song, Bernhard van der Sluis, Yicong Lin',
    author_email='678270ms@eur.nl, vandersluis@ese.eur.nl, yc.lin@vu.nl',
    description = ("The PyTimeVar package offers state-of-the-art estimation and statistical inference methods for time series regression models with flexible trends and/or time- varying coefficients."),
    python_requires=">=3.9",
    keywords = 'time-varying, bootstrap, nonparametric estimation, filtering',
    # packages=['PyTimeVar', 'PyTimeVar.bhpfilter', 'PyTimeVar.locallinear','PyTimeVar.kalman', 'PyTimeVar.powerlaw', 'PyTimeVar.gas', 'PyTimeVar.datasets', 'tests'],
    packages = setuptools.find_packages(where='PyTimeVar'),
    package_dir={"": "PyTimeVar"},
    install_requires=['numpy', 'pandas', 'matplotlib',
                      'scipy', 'statsmodels', 'tqdm'],
    long_description=read('README.md'),
    url = "https://github.com/bpvand/PyTimeVar",
    license = 'GPLv3+',
    classifier=['License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',],
    include_package_data=True,
    package_data={"PyTimeVar.datasets": ["**/*.csv"],}
)
