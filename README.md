# suplearn-clone-detection

[![CircleCI](https://circleci.com/gh/tuvistavie/suplearn-clone-detection.svg?style=svg&circle-token=738ac3f3e6453f2beef09c2bf1a2e72d2a959ee0)](https://circleci.com/gh/tuvistavie/suplearn-clone-detection)

## Setup

```
pip install -r requirements.txt
python setup.py develop
```

Note that Tensorflow needs to be installed separately using [these steps][tensorflow-install]

## Configuration

First, copy [config.yml.example](./config.yml.example) to `config.yml`

```
cp config.yml.example config.yml
```

Then, modify the content of `config.yml`. The configuration file is
self-documented and the most important parameters we used can be found in the paper.

## Dataset

We train our model using a dataset with data extracted from the competitive programming website AtCoder: https://atcoder.jp.
The dataset can be downloaded as an SQLite3 database : [java-python-clones.db.gz][cross-language-clones-db].
You will most likely need to decompress the database before using it.
We also provide the raw data as a tarball : [java-python-clones.tar.gz][cross-language-clones-tar]
The database contains both the text representation and the AST representation
of the source code. All the data is in the `submissions` table. We describe
the different rows of the table below.

Name | Type | Description
-----|------|------------
id | INTEGER | Primary key for the submission
url | VARCHAR(255) | URL of the problem on AtCoder
contest_type | VARCHAR(64) | Contest type on AtCoder (beginner or regular)
contest_id | INTEGER | Contest ID on AtCoder
problem_id | INTEGER | Problem ID on AtCoder
problem_title | VARCHAR(255) | Problem title on AtCoder (usually in Japanese)
filename | VARCHAR(255) | Original path of the file
language | VARCHAR(64) | Full name of the language used
language_code | VARCHAR(64) | Short name of the language used
source_length | INTEGER | Source length in bytes
exec_time | INTEGER | Execution time in ms
tokens_count | INTEGER | Number of tokens in the source
source | TEXT | Source code of the submission
ast | TEXT | JSON encoded AST representation of the source code

The database also contains a `samples` table which should be populated
using the `suplearn-clone generate-dataset` command.

## Usage

The model should already be configured in `config.yml` to use the following steps.

### Generating training samples

Before training the model, the clones pair for training/cross-validation/test must first be generated using the following command.

```
suplearn-clone generate-dataset -c /path/to/config.yml
```

### Training the model

Once the data is generated, the model can be trained
by simply using the following command

```
suplearn-clone train -c /path/to/config.yml
```

### Testing the model

The model can be evaulated on test data by using the following command:

```
suplearn-clone evaulate -c /path/to/config.yml -m /path/to/model.h5 --data-type=<dev|test> -o results.json 
```

Note that `config.yml` should be the same file as the one used for training.

## Using pre-trained embeddings

Pre-trained embeddings can be used by using the `model.languages.n.embeddings`
setting in the configuration file.
This repository does not provide any functionality to train emebddings.
Please check the [bigcode-tools][bigcode-tools] repository for the instructions
on how to train embeddings.


[tensorflow-install]: https://www.tensorflow.org/install
[cross-language-clones-db]: https://static.perez.sh/uploads/2019/cross-language-clone-detection/java-python-clones.db.gz
[cross-language-clones-tar]: https://static.perez.sh/uploads/2019/cross-language-clone-detection/java-python-clones.tar.gz
[bigcode-tools]: https://github.com/danhper/bigcode-tools
