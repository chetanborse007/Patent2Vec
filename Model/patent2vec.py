# -*- coding: utf-8 -*-

"""
@File:           patent2vec.py
@Description:    This is a module for generating document embedding for patents.
                 This application,
                     1. Creates Patent2Vec model
                     2. Initializes Patent2Vec model's weights
                        with pre-trained model
                     3. Trains Patent2Vec model
                     4. Infers document embedding for a new patent document
                     5. Predict document embeddings for the collection of patent documents
                     6. Saves/Loads Patent2Vec model
                     7. Evaluates Patent2Vec model
                     8. Tunes the vocabulary size
                     9. Saves document embeddings to database
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
import time
import logging
import datetime

import numpy as np

from scipy.stats import zscore

from gensim import utils
from gensim.models import Doc2Vec

from Configuration import config

from Utils.exceptions import PathNotFoundError
from Utils.exceptions import ModelNotFoundError

from Utils.database import Database


# Set logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s [%(levelname)s] %(message)s',)
log = logging.getLogger("Patent2Vec")


# Global variables
PATENT2VEC_MODEL_PATH = config.PATENT2VEC_MODEL_PATH
PATENT2VEC_MODEL = config.PATENT2VEC_MODEL
PATENT_EMBEDDING = config.PATENT_EMBEDDING
PATENT_LABEL = config.PATENT_LABEL
PATENT_CATEGORY = config.PATENT_CATEGORY
L2_NORMALIZED_PATENT_EMBEDDING = config.L2_NORMALIZED_PATENT_EMBEDDING
STANDARDIZED_PATENT_EMBEDDING = config.STANDARDIZED_PATENT_EMBEDDING

WORD2VEC_BENCHMARK_DATA = config.WORD2VEC_BENCHMARK_DATA


class Patent2Vec(object):
    """
    This is a class for generating document embedding for patents.

    Note: This implementation is based on Doc2Vec implementation.

    This class,
        1. Creates Patent2Vec model
        2. Initializes Patent2Vec model's weights with pre-trained model
        3. Trains Patent2Vec model
        4. Infers document embedding for a new patent document
        5. Predict document embeddings for the collection of patent documents
        6. Saves/Loads Patent2Vec model
        7. Evaluates Patent2Vec model
        8. Tunes the vocabulary size
        9. Saves document embeddings to database
    """

    def __init__(self,
                 dm=1,
                 dm_mean=1,
                 dm_concat=0,
                 min_word_count=5,
                 max_vocab_size=None,
                 size=500,
                 context_window_size=8,
                 downsampling=1e-5,
                 hs=0,
                 negative=2,
                 iter=50,
                 workers=4,
                 use_less_memory=False,
                 docvecs_mapfile=None):
        self.dm = dm
        self.dm_mean = dm_mean
        self.dm_concat = dm_concat
        self.min_word_count = min_word_count
        self.max_vocab_size = max_vocab_size
        self.size = size
        self.context_window_size = context_window_size
        self.downsampling = downsampling
        self.hs = hs
        self.negative = negative
        self.iter = iter
        self.workers = workers
        self.use_less_memory = use_less_memory
        self.docvecs_mapfile = docvecs_mapfile
        self.model = None

    def build(self, patents):
        """
        Create a Patent2Vec model and build a vocabulary.
        """
        log.info("Building Patent2Vec model")

        # Create a Patent2Vec model
        self.model = Doc2Vec(dm=self.dm,
                             dm_mean=self.dm_mean,
                             dm_concat=self.dm_concat,
                             min_count=self.min_word_count,
                             max_vocab_size=self.max_vocab_size,
                             size=self.size,
                             window=self.context_window_size,
                             sample=self.downsampling,
                             hs=self.hs,
                             negative=self.negative,
                             iter=self.iter,
                             workers=self.workers,
                             docvecs_mapfile=self.docvecs_mapfile)

        # Build a vocabulary of ngrams for a given corpus
        if self.use_less_memory:
            self.model.build_vocab(patents)
        else:
            self.model.build_vocab(patents.to_array())

    def intersect_with_pretrained_embedding(self,
                                            pretrained_word2vec,
                                            binary=False):
        """
        Intersect the vocabulary of ngrams with pre-trained word/concept
        embeddings.

        Note: No new words/concepts are added to the existing vocabulary,
              but intersecting words/concepts adopt the pre-trained
              word/concept embedding's weights and non-intersecting
              words/concepts are left alone.
        """
        log.info("Intersecting vocabulary with pre-trained word/concept embeddings")
        self.model.intersect_word2vec_format(pretrained_word2vec, binary=binary)

    def reuse_from(self, model):
        """
        Reuse shareable structures from other Patent2Vec model.
        """
        self.model.reset_from(model)

    def train(self,
              patents,
              alpha=0.1,
              min_alpha=0.0001,
              passes=10,
              fixed_alpha=False):
        """
        Train Patent2Vec model.

        Note: For rigorous training, set 'passes' > 1.

        There are two training approaches (for every pass),
            1. Flexible learning rate ->
                   Provide minimum & maximum learning rates and
                   let the gensim package adjusts it.
            2. Fixed learning rate ->
                   Provide minimum & maximum learning rates;
                   both must be same and do not let the gensim adjusts
                   the learning rate.
        """
        log.info("Training Patent2Vec model")

        # Compute delta, i.e. change in learning rate after every pass.
        delta = (alpha - min_alpha) / passes

        log.info("START %s", str(datetime.datetime.now()))
        # Train/Re-train model for a given number of passes
        for i in range(passes):
            # Shuffle patent documents
            patents.shuffle()

            # If user chooses fixed learning rate,
            # then 'alpha' and 'min_alpha' should be same;
            # otherwise let the gensim adjusts learning rate
            # after each epoch/iteration.
            self.model.alpha = alpha
            if fixed_alpha:
                self.model.min_alpha = alpha
            else:
                self.model.min_alpha = min_alpha

            # Train Patent2Vec model for the given number of iterations
            # as specified by 'self.iter'
            start_time = time.time()
            if self.use_less_memory:
                self.model.train(patents,
                                 total_examples=len(patents),
                                 epochs=self.iter)
            else:
                self.model.train(patents.get_corpus(),
                                 total_examples=len(patents),
                                 epochs=self.iter)
            end_time = time.time()

            log.debug("Pass(%d): Completed at alpha %f", i+1, alpha)
            log.debug("Pass(%d): Time elapsed = %f", i+1, (end_time-start_time))

            # Lower maximum learning rate's value for next pass
            alpha -= delta
        log.info("END %s", str(datetime.datetime.now()))

    def evaluate(self):
        """
        Evaluate Patent2Vec model.
        """
        log.info("Evaluating Patent2Vec model")

        if not os.path.exists(WORD2VEC_BENCHMARK_DATA):
            raise PathNotFoundError("%s: Evaluation dataset does not exist!"
                                    % WORD2VEC_BENCHMARK_DATA.rsplit(os.sep, 1)[1])

        # Evaluate Patent2Vec model
        accuracy = self.model.accuracy(WORD2VEC_BENCHMARK_DATA)

        # Find correct and incorrect predictions
        correct = len(accuracy[-1]['correct'])
        incorrect = len(accuracy[-1]['incorrect'])
        total = correct + incorrect

        # Calculate correct and incorrect predictions' percentage
        percentage = lambda x: (x / total) * 100

        log.info("Total: %d,  Correct: %0.2f%%,  Incorrect: %0.2f%%",
                 total, percentage(correct), percentage(incorrect))

    def infer(self, document, alpha=0.1, min_alpha=0.0001, steps=100):
        """
        Infer a document embedding for a given patent document.
        """
        return self.model.infer_vector(document,
                                       alpha=alpha,
                                       min_alpha=min_alpha,
                                       steps=steps)

    def predict(self,
                documents,
                alpha=0.1,
                min_alpha=0.0001,
                steps=100,
                save=True,
                database=None,
                table_name=None,
                save_patent_category=True,
                prepend_document_category=False):
        """
        Predict document embeddings.
        """
        log.info("Predicting document embeddings")

        if save and database is None:
            raise ValueError("'database' not defined!")

        if save and table_name is None:
            raise ValueError("'table_name' not defined!")

        # Predict document embeddings
        tags = []
        embeddings = []
        for document in documents:
            embedding = self.infer(document.words[0],
                                   alpha=alpha,
                                   min_alpha=min_alpha,
                                   steps=steps)
            embeddings.append(embedding)
            tags.append(document.tags[0])

        # Insert predicted document embeddings into database
        if save:
            for i, embedding in enumerate(embeddings):
                patent_name = self._get_document_label(tags[i],
                                                       prepend_document_category)
                embedding = " ".join(map(str, embedding))
                if save_patent_category:
                    patent_category = self._get_document_category(tags[i])
                else:
                    patent_category = "UNKNOWN"

                record = [("PatentName", patent_name),
                          ("DocumentEmbedding", embedding),
                          ("PatentCategory", patent_category)]

                database.insert(table=table_name, record=record)

        return (tags, embeddings)

    def save(self, model=None, path=None):
        """
        Save Patent2Vec model.
        """
        log.info("Saving Patent2Vec model")

        if model is None:
            model = PATENT2VEC_MODEL.rsplit(os.sep, 1)[1]

        if path is None:
            path = PATENT2VEC_MODEL_PATH

        if not os.path.exists(path):
            raise PathNotFoundError("Path does not exist: %s" % path)

        self.model.save(os.path.join(path, model))

    def load(self, model):
        """
        Load Patent2Vec model.
        """
        log.info("Loading Patent2Vec model")

        if not os.path.exists(model):
            raise PathNotFoundError("Patent2Vec model does not exist: %s"
                                    % model)

        self.model = Doc2Vec.load(model)

    def clean(self):
        """
        Clean temporary generated data.
        """
        log.info("Cleaning temporary generated data")
        self.model.delete_temporary_training_data()

    def generate_l2_normalized_embeddings(self):
        """
        Generate L2 normalized document embeddings.
        """
        self.model.docvecs.init_sims()

    def standardize_embeddings(self, document_embeddings, rows, columns):
        """
        Standardize document embeddings.
        """
        path = STANDARDIZED_PATENT_EMBEDDING.rsplit(os.sep, 1)[0]

        if not os.path.exists(path):
            raise PathNotFoundError("Path does not exist: %s" % path)

        standardized_patent_embeddings = np.memmap(STANDARDIZED_PATENT_EMBEDDING,
                                                   dtype='float32',
                                                   mode='w+',
                                                   shape=(rows, columns))

        standardized_patent_embeddings[:] = np.array(zscore(document_embeddings))[:]

        return standardized_patent_embeddings

    def save_document_embeddings(self,
                                 document_embeddings=None,
                                 rows=None,
                                 columns=500,
                                 database=None,
                                 table_name=None,
                                 save_patent_category=True,
                                 prepend_document_category=False):
        """
        Save document embeddings to database.
        """
        log.info("Saving document embeddings")

        if document_embeddings is None:
            document_embeddings = PATENT_EMBEDDING

        if not os.path.exists(document_embeddings):
            raise PathNotFoundError("Path does not exist: %s"
                                    % document_embeddings)

        if rows is None:
            raise ValueError("'rows' not defined!")

        if database is None:
            raise ValueError("'database' not defined!")

        if table_name is None:
            raise ValueError("'table_name' not defined!")

        # Create a memory map with document embeddings for reducing load on RAM
        embeddings = np.memmap(document_embeddings,
                               dtype='float32',
                               mode='r',
                               shape=(rows, columns))

        # Insert document embedding records into database
        for i, embedding in enumerate(embeddings):
            doctag = self.model.docvecs.index_to_doctag(i)

            patent_name = self._get_document_label(doctag,
                                                   prepend_document_category)
            embedding = " ".join(map(str, embedding))
            if save_patent_category:
                patent_category = self._get_document_category(doctag)
            else:
                patent_category = "UNKNOWN"

            record = [("PatentName", patent_name),
                      ("DocumentEmbedding", embedding),
                      ("PatentCategory", patent_category)]

            database.insert(table=table_name, record=record)

    def _get_document_label(self, doctag, prepend_document_category=False):
        """
        Get document label.
        """
        document_name = doctag.rsplit(os.sep, 1)[1]
        document_name = document_name.rsplit('.', 1)[0]

        if prepend_document_category:
            document_category = doctag.rsplit(os.sep, 2)[1]
        else:
            document_category = ""

        if document_category:
            document_label = document_category + "." + document_name
        else:
            document_label = document_name

        return document_label

    def _get_document_category(self, doctag, description=None):
        """
        Get document category.
        """
        document_category = doctag.rsplit(os.sep, 2)[1]

        if description == "20ng_6categories":
            if document_category in ["comp.graphics",
                                     "comp.os.ms-windows.misc",
                                     "comp.sys.ibm.pc.hardware",
                                     "comp.sys.mac.hardware",
                                     "comp.windows.x"]:
                return "computer"
            if document_category in ["talk.politics.misc",
                                     "talk.politics.guns",
                                     "talk.politics.mideast"]:
                return "politics"
            if document_category in ["talk.religion.misc",
                                     "alt.atheism",
                                     "soc.religion.christian"]:
                return "religion"
            if document_category in ["sci.crypt",
                                     "sci.electronics",
                                     "sci.med",
                                     "sci.space"]:
                return "science"
            if document_category in ["misc.forsale"]:
                return "forsale"
            if document_category in ["rec.autos",
                                     "rec.motorcycles",
                                     "rec.sport.baseball",
                                     "rec.sport.hockey"]:
                return "rec"

        return document_category

    def tune_vocab_size(self, patents, min_count_range=(0, 50)):
        """
        Function for tuning the vocabulary size.
        """
        if self.model is None:
            raise ModelNotFoundError("Patent2Vec model not found!")

        # Scan vocabulary across entire corpus
        if self.use_less_memory:
            self.model.scan_vocab(patents)
        else:
            self.model.scan_vocab(patents.to_array())

        # Find out the vocabulary size for different minimum token frequency
        for i in range(min_count_range[0], min_count_range[1]):
            report = self.model.scale_vocab(min_count=i,
                                            dry_run=True,
                                            keep_raw_vocab=False)

            if self.hs:
                vocab_size = report['memory']['vocab'] / 700
            else:
                vocab_size = report['memory']['vocab'] / 500

            log.info("For min_count(%d), vocab_size = %d", i, vocab_size)


class ConcatenatedPatent2Vec(object):
    """
    Wrapper class to produce concatenated Patent2Vec models.
    """

    def __init__(self, models):
        self.models = models
        if hasattr(models[0], 'docvecs'):
            self.docvecs = ConcatenatedDocvecs([model.docvecs for model in models])

    def __getitem__(self, token):
        return np.concatenate([model[token] for model in self.models])

    def infer_vector(self, document, alpha=0.1, min_alpha=0.0001, steps=5):
        return np.concatenate([model.infer_vector(document, alpha, min_alpha, steps)
                               for model in self.models])

    def train(self, ignored):
        pass


class ConcatenatedDocvecs(object):

    def __init__(self, models):
        self.models = models

    def __getitem__(self, token):
        return np.concatenate([model[token] for model in self.models])


class AvgPatent2Vec(object):
    """
    This is a class for generating document embedding for patents.

    Note: This implementation averages token embeddings for all tokens
          in a document for generating a corresponding document embedding.

    This class,
        1. Generates document embedding for a patent document
        2. Saves document embeddings to database
    """

    def __init__(self, embedding_size=500):
        self.embedding_size = embedding_size

    def train(self, patents):
        """
        Generates document embedding for a patent document.
        """
        log.info("Training AvgPatent2Vec model")

        if not os.path.exists(PATENT_EMBEDDING.rsplit(os.sep, 1)[0]):
            raise PathNotFoundError("Path does not exist: %s"
                                    % PATENT_EMBEDDING.rsplit(os.sep, 1)[0])

        if not os.path.exists(PATENT_LABEL.rsplit(os.sep, 1)[0]):
            raise PathNotFoundError("Path does not exist: %s"
                                    % PATENT_LABEL.rsplit(os.sep, 1)[0])

        if not os.path.exists(PATENT_CATEGORY.rsplit(os.sep, 1)[0]):
            raise PathNotFoundError("Path does not exist: %s"
                                    % PATENT_CATEGORY.rsplit(os.sep, 1)[0])

        self.total_docs = len(patents)

        doc_embeddings = np.memmap(PATENT_EMBEDDING,
                                   dtype='float32',
                                   mode='w+',
                                   shape=(self.total_docs, self.embedding_size))

        doc_labels = np.memmap(PATENT_LABEL,
                               dtype="object",
                               mode='w+',
                               shape=(self.total_docs,))

        doc_categories = np.memmap(PATENT_CATEGORY,
                                   dtype="object",
                                   mode='w+',
                                   shape=(self.total_docs,))

        # Find document embedding by averaging token embeddings
        # for all tokens within a document
        for i, patent in enumerate(patents):
            word_embeddings = patent.words[1]

            word_count = 0
            doc_embedding = np.zeros((500,), dtype=np.float32)
            for word_embedding in word_embeddings:
                if isinstance(word_embedding, np.ndarray):
                    doc_embedding += word_embedding
                    word_count += 1

            if word_count != 0:
                doc_embedding /= word_count

            # Document embedding
            doc_embeddings[i] = doc_embedding

            # Document label
            doc_label = patent.tags[0]
            doc_label = doc_label.rsplit(os.sep, 1)[1]
            doc_label = doc_label.rsplit('.', 1)[0]
            doc_labels[i] = doc_label

            # Document category
            doc_category = patent.tags[0]
            doc_category = doc_category.rsplit(os.sep, 2)[1]
            doc_categories[i] = doc_category

    def save_document_embeddings(self,
                                 document_embeddings=None,
                                 doc_labels=None,
                                 doc_categories=None,
                                 rows=None,
                                 columns=500,
                                 database=None,
                                 table_name=None,
                                 save_patent_category=True):
        """
        Save document embeddings to database.
        """
        log.info("Saving document embeddings")

        if document_embeddings is None:
            document_embeddings = PATENT_EMBEDDING

        if doc_labels is None:
            doc_labels = PATENT_LABEL

        if doc_categories is None:
            doc_categories = PATENT_CATEGORY

        if not os.path.exists(document_embeddings):
            raise PathNotFoundError("Path does not exist: %s"
                                    % document_embeddings)

        if not os.path.exists(doc_labels):
            raise PathNotFoundError("Path does not exist: %s"
                                    % doc_labels)

        if not os.path.exists(doc_categories):
            raise PathNotFoundError("Path does not exist: %s"
                                    % doc_categories)

        if rows is None:
            rows = self.total_docs

        if columns is None:
            columns = self.embedding_size

        if database is None:
            raise ValueError("'database' not defined!")

        if table_name is None:
            raise ValueError("'table_name' not defined!")

        # Create a memory map with document embeddings for reducing load on RAM
        embeddings = np.memmap(document_embeddings,
                               dtype='float32',
                               mode='r',
                               shape=(rows, columns))

        # Create a memory map with document labels for reducing load on RAM
        labels = np.memmap(doc_labels,
                           dtype="object",
                           mode='r',
                           shape=(rows,))

        # Create a memory map with document categories for reducing load on RAM
        categories = np.memmap(doc_categories,
                               dtype="object",
                               mode='r',
                               shape=(rows,))

        # Insert document embedding records into database
        for i, embedding in enumerate(embeddings):
            patent_name = labels[i]
            embedding = " ".join(map(str, embedding))
            if save_patent_category:
                patent_category = self._get_document_category(categories[i])
            else:
                patent_category = "UNKNOWN"

            record = [("PatentName", patent_name),
                      ("DocumentEmbedding", embedding),
                      ("PatentCategory", patent_category)]

            db.insert(table=table_name, record=record)

    def _get_document_category(self, doctag, description=None):
        """
        Get document category.
        """
        document_category = doctag

        if description == "20ng_6categories":
            if document_category in ["comp.graphics",
                                     "comp.os.ms-windows.misc",
                                     "comp.sys.ibm.pc.hardware",
                                     "comp.sys.mac.hardware",
                                     "comp.windows.x"]:
                return "computer"
            if document_category in ["talk.politics.misc",
                                     "talk.politics.guns",
                                     "talk.politics.mideast"]:
                return "politics"
            if document_category in ["talk.religion.misc",
                                     "alt.atheism",
                                     "soc.religion.christian"]:
                return "religion"
            if document_category in ["sci.crypt",
                                     "sci.electronics",
                                     "sci.med",
                                     "sci.space"]:
                return "science"
            if document_category in ["misc.forsale"]:
                return "forsale"
            if document_category in ["rec.autos",
                                     "rec.motorcycles",
                                     "rec.sport.baseball",
                                     "rec.sport.hockey"]:
                return "rec"

        return document_category
