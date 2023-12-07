![example workflow](https://github.com/fedorbondar/research-keywords/actions/workflows/run-tests.yml/badge.svg)

# research-keywords

Project for working with text descriptions of tests in Markdown format. 
It contains tools for various ways of comparing texts and searching for keywords.
They combine classic NLP approaches with the power of `transformers`.

**What is a keyword in this context?** Generally, "keyword" is an approach in testing 
when some logical block is designated by a key phrase and need not be 
deciphered in the text of the test.

**For example:** we use the keyword "register" and mean by this a set of actions that 
need to be performed in the application in order to register and gain access. 
We record this set of actions in the keyword description, and in the test 
we simply use "register".

This project code is fully written in Python 3.9 and convenient to use as a 
console utility.
Usage examples can be easily derived from tests, feel free to look through them.
Synthetic examples of real test cases with respect to original design (can be found 
[here](https://github.com/fedorbondar/research-keywords/tree/main/src/case-examples))
are also friends of yours.

## Features

* Text comparison

There are various methods of test case comparison available. Once text is preprocessed
and prepared with either ngrams, random sentence split or RAKE algorithm, it then can be 
vectorized with Tfidf or BERT.

* Keyword detection 

Both format-specific keyword detection methods and generalized search methods 
are available.

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