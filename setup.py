""" sc2ai module setup script """
from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="SC2AI",
    version="0.0.1",
    description="Brown's Starcraft 2 Reinforcement Learning Training Framework",
    author="Brown University",
    author_email="jun_ki_lee@brown.edu",
    long_description=long_description,
    keywords="Brown Starcraft AI",
    url="https://github.com/junkilee/sc2ai",
    packges=['s2cai'],
    install_requires=[],
    entry_points={},
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: Mac OS X",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
        "Intended Audience :: Science/Research"
    ]
)