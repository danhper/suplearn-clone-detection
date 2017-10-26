# suplearn-clone-detection

[![CircleCI](https://circleci.com/gh/tuvistavie/suplearn-clone-detection.svg?style=svg&circle-token=738ac3f3e6453f2beef09c2bf1a2e72d2a959ee0)](https://circleci.com/gh/tuvistavie/suplearn-clone-detection)

## Setup

```
pip install -r requirements.txt
python setup.py develop
```

## Sample usage

```python
from suplearn_clone_detection.data_generator import DataGenerator

generator = DataGenerator("submissions.json", "asts.json", "asts.txt")
while True:
  asts, labels = generator.next_batch(256)
  if not asts:
    break
  # labels: list of 0 or 1
  # asts: list of AST pairs
```
