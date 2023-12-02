![example workflow](https://github.com/fedorbondar/research-keywords/actions/workflows/run-tests.yml/badge.svg)

# research-keywords

Project for working with text descriptions of tests in Markdown format.

## Features

* Text comparison
* Keyword detection 

## Requirements

See [here](https://github.com/fedorbondar/research-keywords/blob/main/requirements.txt).

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

## References

* "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation" by N. Reimers, I. Gurevych
  ([source](https://arxiv.org/abs/2004.09813))
* "Automatic Keyword Extraction from Individual Documents" by S. Rose, D. Engel, N. Cramer W. Cowley
  ([source](https://www.researchgate.net/publication/227988510_Automatic_Keyword_Extraction_from_Individual_Documents))
* "Python implementation of the Rapid Automatic Keyword Extraction algorithm using NLTK" by Vishwas B. Sharma
  ([source](https://csurfer.github.io/rake-nltk/_build/html/index.html))

## License

[MIT License](https://github.com/fedorbondar/research-keywords/blob/main/LICENSE)