# suplearn-clone-detection

[![CircleCI](https://circleci.com/gh/tuvistavie/suplearn-clone-detection.svg?style=svg&circle-token=738ac3f3e6453f2beef09c2bf1a2e72d2a959ee0)](https://circleci.com/gh/tuvistavie/suplearn-clone-detection)

## Setup

```
pip install -r requirements.txt
python setup.py develop
```

## Configuration

See [config.yml.example](./config.yml.example) for a sample configuration file.
The file should be copied as `config.yml` to be used automatically.

## Usage

```
$ ./bin/suplearn-clone -h
usage: suplearn-clone [-h] [-q] [--debug]
                      {train,generate-data,evaluate,predict,show-results} ...

positional arguments:
  {train,generate-data,evaluate,predict,show-results}
    train               Train the model
    generate-data       Generate data for evaluating model
    evaluate            Evaluate the model
    predict             Predict files
    show-results        Show formatted results

optional arguments:
  -h, --help            show this help message and exit
  -q, --quiet           reduce output
  --debug               enables debug
```
