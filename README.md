# Setup
## For: Python 3.5

## Mongo Install & Run
- brew update
- brew install mongodb
- mkdir -p /data/db
- sudo mongod

## Anaconda Install
- https://repo.continuum.io/archive/Anaconda3-4.2.0-MacOSX-x86_64.pkg

## Anaconda Environment + Python Packages Install
- conda create --name mslc scipy numpy matplotlib tornado scikit-learn pymongo

## Tornado/Application Run
- cd \<XcodeProjectFolder of Assignment6\>/tornado_bare-sklearn_example
- source activate mslc
- python tornado_scikit_learn.py
