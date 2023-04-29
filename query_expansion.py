import gensim.downloader as api
import numpy as np
from PyDictionary import PyDictionary
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import string
import json


def expand_with_synonyms(query):
    tokens = query.split()

    # Find synonyms for each token
    dictionary=PyDictionary()
    new_tokens = []
    for token in tokens:
        synonyms = []
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        if synonyms:
            new_token = dictionary.meaning(token)
            if new_token:
                new_token = new_token.popitem()[1][0]
            else:
                new_token = synonyms[0]
        else:
            new_token = token
        new_tokens.append(new_token)

    # Create new query with expanded terms
    new_query = " ".join(new_tokens)
    return new_query