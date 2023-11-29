![example workflow](https://github.com/fedorbondar/research-keywords/actions/workflows/run-tests.yml/badge.svg)

# research-keywords

Project for working with text descriptions of tests in Markdown format.

## Features

* Text comparison
* Keyword detection 

## Requirements

See [here](../research-keywords/requirements.txt).

Best way to get all required packages at once is running the following line:

```shell
pip install -r requirements.txt
```

After installation of `nltk` you might also need to execute the 
following in python:
```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
```

## Installation 

* Clone this repo
* Make sure you've satisfied the [requirements](#requirements)
* For text comparison run the line like:

```shell
python main.py path_to_cases_folder path/new_case.md [silent | print | log]
```

The last argument is optional and set `silent` by default.
