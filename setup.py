import os
import runpy
from setuptools import setup, find_packages

# Get version
cwd = os.path.abspath(os.path.dirname(__file__))
versionpath = os.path.join(cwd, 'fpsim', 'version.py')
version = runpy.run_path(versionpath)['__version__']

# Get the documentation
with open(os.path.join(cwd, 'README.rst'), "r") as f:
    long_description = f.read()

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: Other/Proprietary License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python :: 3.10",
]

setup(
    name="fpsim",
    version=version,
    author="Michelle O'Brien, Annie Valente, Cliff Kerr, Sam Buxton, Daniel Klein, Marita Zimmermann, Emily Driano",
    author_email="info@fpsim.org",
    description="FPsim: Family Planning Simulator",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url='http://fpsim.org',
    keywords=["family planning", "women's health", "agent-based model", "simulation"],
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'numba',
        'scipy',
        'starsim>=3.1.0',
        'pandas>=1.3', 
        'sciris>=2.1.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.9',
        'optuna',
        'plotnine',
        'pyarrow',
        'pyyaml',
    ],
)
