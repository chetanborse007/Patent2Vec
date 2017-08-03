# -*- coding: utf-8 -*-

"""
@File:           preprocessor.py
@Description:    This is a module for preprocessing patent documents.
                 This application,
                     1. Tokenizes patent document
                     2. Removes stop words
                     3. Part of speech tagging
                     4. WordNet Lemmatization
                     5. Conceptualizes patent document
                     6. Generates ngrams for a given patent document
@Author:         Chetan Borse
@EMail:          chetanborse2106@gmail.com
@Created_on:     04/05/2017
@License         Copyright [2017] [Chetan Borse]

                 Licensed under the Apache License, Version 2.0 (the "License");
                 you may not use this file except in compliance with the License.
                 You may obtain a copy of the License at

                 http://www.apache.org/licenses/LICENSE-2.0

                 Unless required by applicable law or agreed to in writing,
                 software distributed under the License is distributed on an
                 "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
                 either express or implied.
                 See the License for the specific language governing permissions
                 and limitations under the License.
@python_version: 3.5
===============================================================================
"""


import os
import re
import sys
import json
import random
import codecs
import logging

import numpy as np

from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tag.mapping import map_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN
from nltk.corpus.reader.wordnet import VERB
from nltk.corpus.reader.wordnet import ADJ
from nltk.corpus.reader.wordnet import ADV

from gensim import utils
from gensim.models.doc2vec import TaggedDocument

from Configuration import config

from Utils.exceptions import PathNotFoundError
from Utils.exceptions import ModelNotFoundError

from TextConceptualizer.conceptualizer import Conceptualizer


# Set logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s [%(levelname)s] %(message)s',)
log = logging.getLogger("PatentDocument")


# Global variables
PATENT_FIELDS = config.PATENT_FIELDS
SAMPLED_DATA_PATH = config.SAMPLED_DATA_PATH
TRAINING_DATA = config.TRAINING_DATA
TESTING_DATA = config.TESTING_DATA

STANFORD_POS_TAGGER = config.STANFORD_POS_TAGGER
STANFORD_JAR = config.STANFORD_JAR

CONCEPTUALIZER = config.CONCEPTUALIZER
CONCEPTUALIZER_WORD2VEC_FORMAT = config.CONCEPTUALIZER_WORD2VEC_FORMAT


class PatentDocument(object):
    """
    This is a class for preprocessing patent documents.

    This class,
        1. Tokenizes patent document
        2. Removes stop words
        3. Part of speech tagging
        4. WordNet Lemmatization
        5. Conceptualizes patent document
        6. Generates ngrams for a given patent document
    """

    # Different encoding types supported
    SOURCE_ENCODING = ["utf-8", "iso8859-1", "latin1"]

    if sys.version > '3':
        CONTROL_CHAR = [chr(0x85)]
    else:
        CONTROL_CHAR = [unichr(0x85)]

    # Punctuation symbols
    PUNCTUATION = ['.', '"', ',', '(', ')', '{', '}', '[', ']', '<', '>', '!',
                   '?', ';', ':']

    # Default stop words collection
    STOPWORDS = set(stopwords.words("english"))

    # Tagset mapping
    TAGSET = {'NOUN': NOUN,
              'VERB': VERB,
              'ADJ': ADJ,
              'ADV': ADV}

    def __init__(self,
                 source,
                 extension=".xml",
                 remove_stopwords=True,
                 enable_pos_tagging=True,
                 enable_lemmatization=True,
                 token_only=False,
                 use_conceptualizer=True,
                 concept_only=False,
                 require_embedding=False,
                 transform_conceptualizer=False,
                 enable_sampling=False,
                 train_ratio=1.0,
                 test_ratio=0.0,
                 java_options='-mx4096m'):
        self.source = source
        self.extension = extension
        self.remove_stopwords = remove_stopwords
        self.enable_pos_tagging = enable_pos_tagging
        self.enable_lemmatization = enable_lemmatization
        self.token_only = token_only
        self.use_conceptualizer = use_conceptualizer
        self.concept_only = concept_only
        self.require_embedding = require_embedding
        self.transform_conceptualizer = transform_conceptualizer
        self.enable_sampling = enable_sampling
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.java_options = java_options
        self.corpus = None

        # List all source documents with provided extension
        self.documents, self.test_documents = self._list_documents(source,
                                                                   extension,
                                                                   save=True)

        # Part of Speech Tagger
#         if enable_pos_tagging:
#             self.pos_tagger = StanfordPOSTagger(STANFORD_POS_TAGGER,
#                                                 path_to_jar=STANFORD_JAR,
#                                                 encoding='utf8',
#                                                 verbose=False,
#                                                 java_options=java_options)

        # WordNet Lemmatizer
        if enable_lemmatization:
            self.lemmatizer = WordNetLemmatizer()

        # If 'use_conceptualizer' is True,
        # then load conceptualizer module to conceptualize text
        if use_conceptualizer:
            log.info("Loading conceptualizer model: %s",
                     CONCEPTUALIZER.rsplit(os.sep, 1)[1])
            self._load_conceptualizer(CONCEPTUALIZER)

            # Transform conceptualizer model into Word2Vec format
            if transform_conceptualizer:
                log.info("Transforming conceptualizer model into Word2Vec format")
                self._transform_conceptualizer(save_to=CONCEPTUALIZER_WORD2VEC_FORMAT)

    def set_token_only(self, token_only=False):
        """
        Set 'token_only' flag to True, if only tokens are required;
        otherwise set it to False in order to get TaggedDocuments.
        """
        self.token_only = token_only

    def get_document_list(self):
        """
        Get patent documents list.
        """
        return self.documents

    def get_corpus(self):
        """
        Get patent corpus.
        """
        return self.corpus

    def get_preprocessed_document(self, document):
        """
        Get preprocessed patent document.
        """
        if not os.path.exists(document):
            raise PathNotFoundError("%s: Document does not exist!"
                                    % document.rsplit(os.sep, 1)[1])

        for source_encoding in PatentDocument.SOURCE_ENCODING:
            with codecs.open(document, "r", source_encoding) as d:
                try:
                    content = self._read(d)
                except UnicodeDecodeError as e:
                    continue

                preprocessed_content = self._preprocess(content, lowercase=True)

                if self.token_only:
                    return preprocessed_content
                else:
                    return TaggedDocument(preprocessed_content, [document])

        return None

    def get_preprocessed_corpus(self, corpus):
        """
        Get the corpus of preprocessed patent documents.
        """
        preprocessed_corpus = []

        for document in corpus:
            if not os.path.exists(document):
                continue

            for source_encoding in PatentDocument.SOURCE_ENCODING:
                with codecs.open(document, "r", source_encoding) as d:
                    try:
                        content = self._read(d)
                    except UnicodeDecodeError as e:
                        continue

                    preprocessed_content = self._preprocess(content,
                                                            lowercase=True)

                    if self.token_only:
                        preprocessed_corpus.append(preprocessed_content)
                    else:
                        preprocessed_corpus.append(TaggedDocument(preprocessed_content,
                                                                  [document]))

                break

        return preprocessed_corpus

    def __len__(self):
        """
        Return total number of patent documents in corpus.
        """
        return len(self.documents)

    def __iter__(self):
        """
        Returns iterator to iterate over patent documents.
        """
        for document in self.documents:
            for source_encoding in PatentDocument.SOURCE_ENCODING:
                with codecs.open(document, "r", source_encoding) as d:
                    try:
                        content = self._read(d)
                    except UnicodeDecodeError as e:
                        continue

                    preprocessed_content = self._preprocess(content,
                                                            lowercase=True)

                    if self.token_only:
                        yield preprocessed_content
                    else:
                        yield TaggedDocument(preprocessed_content, [document])

                break

    def to_array(self):
        """
        Returns array of patent documents.
        """
        self.corpus = []

        for document in self.documents:
            for source_encoding in PatentDocument.SOURCE_ENCODING:
                with codecs.open(document, "r", source_encoding) as d:
                    try:
                        content = self._read(d)
                    except UnicodeDecodeError as e:
                        continue

                    preprocessed_content = self._preprocess(content,
                                                            lowercase=True)

                    if self.token_only:
                        self.corpus.append(preprocessed_content)
                    else:
                        self.corpus.append(TaggedDocument(preprocessed_content,
                                                          [document]))

                break

        return self.corpus

    def shuffle(self):
        """
        Shuffle patent documents.
        """
        if self.documents:
            np.random.shuffle(self.documents)

    def _list_documents(self, source, extension=".xml", save=False):
        """
        List all patent documents within corpus.
        """
        if not os.path.exists(source):
            raise PathNotFoundError("%s: Source does not exist!"
                                     % source.rsplit(os.sep, 1)[1])

        documents = []
        test_documents = []

        for root, folders, files in os.walk(source):
            for file in files:
                if not file.startswith('.'):
                    if file.endswith(extension):
                        if self.enable_sampling:
                            if random.random() <= self.train_ratio:
                                documents.append(os.path.join(root, file))
                            else:
                                test_documents.append(os.path.join(root, file))
                        else:
                            documents.append(os.path.join(root, file))

        if save:
            if not os.path.exists(SAMPLED_DATA_PATH):
                raise PathNotFoundError("Sampled data path does not exist: %s"
                                         % SAMPLED_DATA_PATH)

            with open(TRAINING_DATA, "w") as f:
                f.write("\n".join(documents))

            with open(TESTING_DATA, "w") as f:
                f.write("\n".join(test_documents))

        return (documents, test_documents)

    def _load_conceptualizer(self, model, log_every=1000000):
        """
        Load conceptualizer model.
        """
        if not os.path.exists(model):
            raise PathNotFoundError("%s: Conceptualizer does not exist!"
                                     % model.rsplit(os.sep, 1)[1])

        # Create conceptualizer's object
        self.conceptualizer = Conceptualizer()

        # Load conceptualizer model
        self.conceptualizer.load(model_path=model, log_every=log_every)

    def _transform_conceptualizer(self, save_to=CONCEPTUALIZER_WORD2VEC_FORMAT):
        """
        Transform conceptualizer model into Word2Vec format.
        """
        if self.conceptualizer.model is None:
            raise ModelNotFoundError("Conceptualizer model not found!")

        # Get concept-embeddings
        concept_embeddings = self._get_concept_embedding()

        with codecs.open(save_to, "w", "utf-8") as m:
            # Vocabulary size
#             vocab_size = len(self.conceptualizer.model)
            vocab_size = len(concept_embeddings)

            # Feature dimension
#             vector_size = len(next(iter(self.conceptualizer.model.values())))
            vector_size = len(next(iter(concept_embeddings.values())))

            # Save header first
            header = str(vocab_size) + " " + str(vector_size)
            m.write(header)

            # Save <Token> => <Concept Embedding> mapping
#             for token, embedding in self.conceptualizer.model.items():
            for token, embedding in concept_embeddings.items():
                token = "_".join(token.split())

                if type(embedding) == str:
                    embedding = embedding.strip()
                else:
                    embedding = " ".join(map(str, embedding))

                m.write("\n" + token + " " + embedding)

    def _get_concept_embedding(self):
        """
        Get the dictionary of <Token> => <Concept Embedding> mapping.
        """
        concept_embeddings = {}

        # For every token in vocabulary,
        for token in self.conceptualizer.vocabulary:
            # This is a concept. Ignore it; we will catch it later.
            if re.findall('id\d+di', token) != []:
                continue

            token_info = self.conceptualizer.all_titles.get(token)

            # This is a normal word. Check to see if it has an embedding.
            if token_info is None:
                embedding = self.conceptualizer.model.get(token)
                if embedding is not None:
                    concept_embeddings[token] = embedding
#                     yield (token, embedding)
            # This is a concept. Get its embedding using the id.
            else:
                concept_embeddings[token] = self.conceptualizer.model[token_info[0]]
#                 yield (token, self.conceptualizer.model[token_info[0]])

        return concept_embeddings

    def _read(self, handle):
        """
        Read the content.
        """
        content = ""

        try:
            # Load JSON data
            data = json.load(handle)
            
            # Parse only the required content
            for k in PATENT_FIELDS:
                try:
                    content += data[k]
                except KeyError as e:
                    continue
                else:
                    content += "\n"
        except ValueError as e:
            # If content is not in JSON format, then read it in raw format
            content = handle.read()
        
        return content

    def _preprocess(self,
                    content,
                    min_len=2,
                    max_len=15,
                    min_ngrams=1,
                    max_ngrams=8,
                    token_pattern=r"(?u)\b\w\w+\b",
                    deaccent=False,
                    lowercase=False,
                    control_char=None,
                    punctuation=None):
        """
        Preprocess patent document.

        This function,
            1. Tokenizes patent document
            2. Removes stop words
            3. Part of speech tagging
            4. WordNet Lemmatization
            5. Conceptualizes patent document
            6. Generates ngrams for a given patent document
        """
        tokens = []

        if control_char is None:
            control_char = PatentDocument.CONTROL_CHAR

        if punctuation is None:
            punctuation = PatentDocument.PUNCTUATION

        # Replace control characters with white-spaces
        for c in control_char:
            content = content.replace(c, ' ')

        # Pad punctuation with spaces on both sides
        for p in punctuation:
            content = content.replace(p, ' ' + p + ' ')

        # If 'use_conceptualizer' is True, then conceptualize document contents
        if self.use_conceptualizer:
            # If 'remove_stopwords' is True, then specify stop-word collection
            if self.remove_stopwords:
                stop_words = PatentDocument.STOPWORDS
            else:
                stop_words = None

            # Conceptualize document contents
            conceptualized_content = self.conceptualizer.conceptualize(input_text=content,
                                                                       ngram_range=(min_ngrams, max_ngrams),
                                                                       stop_words=stop_words,
                                                                       token_pattern=r"(?u)\b\w\w+\b",
                                                                       lowercase=lowercase)

            # If 'concept_only' is True, then discard tokens othern than concepts
            if self.concept_only:
                conceptualized_content = [concept
                                          for concept in conceptualized_content
                                          if concept.title]

            # List of tokens (ngrams)
            tokens = ['_'.join(concept.ngram.split())
                      for concept in conceptualized_content]

            # If 'require_embedding' is True, then extract token embedding
            if self.require_embedding:
                embeddings = [concept.embeddings
                              for concept in conceptualized_content]
        # Else tokenizes document contents and perform simple text processing
        else:
            # Tokenization
            for token in utils.tokenize(content,
                                        lowercase,
                                        deaccent,
                                        errors='ignore'):
                if (min_len <= len(token) <= max_len and 
                        not token.startswith('_')):
                    tokens.append(token)

            # Stop-word removal
            if self.remove_stopwords:
                tokens = [token
                          for token in tokens
                          if token not in PatentDocument.STOPWORDS]

        # NLTK's WordNet Lemmatization
        if self.enable_lemmatization:
            tokens = self._lemmatize(tokens,
                                     unigram_only=True,
                                     delimiter="_")

        # If 'require_embedding' is True, then return (tokens, embeddings)
        if self.require_embedding:
            return (tokens, embeddings)

        return tokens

    def _pos_tag(self, tokens, tagset="universal"):
        """
        NLTK's part of speech tagging to tag the given list of tokens.
        """
#         tagged_tokens = self.pos_tagger.tag(tokens)
        tagged_tokens = pos_tag(tokens, tagset)

#         if tagset:
#             tagged_tokens = [(token, map_tag('en-ptb', tagset, tag))
#                              for (token, tag) in tagged_tokens]

        return tagged_tokens

    def _lemmatize(self, tokens, unigram_only=True, delimiter="_"):
        """
        NLTK's WordNet Lemmatization for lemmatizing the given list of tokens.
        """
        tagged_tokens = self._pos_tag(tokens)

        tagged_tokens = [(token, tag)
                         if tag in PatentDocument.TAGSET
                         else (token, 'NOUN')
                         for (token, tag) in tagged_tokens]

        if unigram_only:
            lemmatized_tokens = [self.lemmatizer.lemmatize(token, pos=PatentDocument.TAGSET[tag])
                                 if len(token.split(delimiter)) == 1
                                 else token
                                 for (token, tag) in tagged_tokens]
        else:
            lemmatized_tokens = [self.lemmatizer.lemmatize(token, pos=PatentDocument.TAGSET[tag])
                                 for (token, tag) in tagged_tokens]

        return lemmatized_tokens
