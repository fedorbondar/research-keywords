from nltk.corpus import stopwords

HEADERS_AND_FORMAT = ['keyword', 'url', 'title', 'name', 'tors',
                      'section', 'id', 'refs', 'main', 'product',
                      'description', 'preconditions', 'content', 'expected', 'step',
                      'https', 'значение', 'описание', 'параметр', 'параметры',
                      'входные', 'выходные']

RUSSIAN_STOPWORDS = stopwords.words('russian')

LOGS_DIRECTORY = __file__[:-12]
