# research-keywords

Project for working with text descriptions of tests in Markdown format.

## Features

* Text comparison
* Keyword detection 

## Requirements

See requirements.txt

After installation of `nltk` you might also need to execute the 
following in python:
```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
```

## Installation 

Clone repo and use the following command:

```shell
python path_to_file/main.py path_to_folder_with_cases path_to_case/new_case.md
```