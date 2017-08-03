# -*- coding: utf-8 -*-

"""
@File:           config.py
@Description:    This is a placeholder for the different configurations.
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
import platform
import multiprocessing


# Cluto binary
CLUTO_VCLUSTER_BINARY = os.path.join(os.getcwd(),
                                     "bin",
                                     "cluto-2.1.2",
                                     platform.system()+"-"+platform.processor(),
                                     "vcluster")


# Maximum CPU cores on a node
CPU_CORE = multiprocessing.cpu_count()


# Source data options
SOURCE_DATASET = os.path.join(os.getcwd(), "data", "source", "Tipster")
SAMPLED_DATA_PATH = os.path.join(os.getcwd(), "output", "sample")
TRAINING_DATA = os.path.join(SAMPLED_DATA_PATH, "train")
TESTING_DATA = os.path.join(SAMPLED_DATA_PATH, "test")


# Benchmark data options
WORD2VEC_BENCHMARK_DATA = os.path.join(os.getcwd(),
                                       "data",
                                       "benchmark",
                                       "doc2vec",
                                       "questions-words.txt")
CLUSTERING_BENCHMARK_DATA = os.path.join(os.getcwd(),
                                         "data",
                                         "benchmark",
                                         "clustering",
                                         "20_newsgroups_preprocessed")
TEST_DOCUMENT = os.path.join(os.getcwd(),
                             "data",
                             "benchmark",
                             "clustering",
                             "20_newsgroups_preprocessed",
                             "rec.autos",
                             "103325")


# Pretrained embedding options
#PRETRAINED_EMBEDDING = os.path.join(os.getcwd(),
#                                    "data",
#                                    "pretrained_embedding",
#                                    "GoogleNews-vectors-negative300.bin")
# PRETRAINED_EMBEDDING = os.path.join(os.getcwd(), "data", "pretrained_embedding", "wiki.en.vec")
PRETRAINED_EMBEDDING = os.path.join(os.getcwd(),
                                    "data",
                                    "pretrained_embedding",
                                    "w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.vec")
#PRETRAINED_EMBEDDING = os.path.join(os.sep,
#                                    "scratch",
#                                    "cborse",
#                                    "clustering",
#                                    "data",
#                                    "word_embedding",
#                                    "w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.vec")


# Conceptualizer options
CONCEPTUALIZER = os.path.join(os.getcwd(),
                              "data",
                              "pretrained_embedding",
                              "w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.bin")
#CONCEPTUALIZER = os.path.join(os.sep,
#                              "scratch",
#                              "cborse",
#                              "w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.bin")
CONCEPTUALIZER_WORD2VEC_FORMAT = os.path.join(os.getcwd(),
                                              "data",
                                              "pretrained_embedding",
                                              "w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.vec")
#CONCEPTUALIZER_WORD2VEC_FORMAT = os.path.join(os.sep,
#                                              "scratch",
#                                              "cborse",
#                                              "clustering",
#                                              "data",
#                                              "word_embedding",
#                                              "w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.vec")


# Patent2Vec options
PATENT2VEC_MODEL_PATH = os.path.join(os.getcwd(), "output", "models")
PATENT2VEC_MODEL = os.path.join(PATENT2VEC_MODEL_PATH, "patent2vec.p2v")

PATENT2VEC_PATH = os.path.join(os.getcwd(), "output", "docvecs")
DOCVECS_MAP = os.path.join(PATENT2VEC_PATH, "patents.map")
PATENT_EMBEDDING = os.path.join(PATENT2VEC_PATH, "patents.map.doctag_syn0")
PATENT_LABEL = os.path.join(PATENT2VEC_PATH, "patents.map.doctag_syn0_labels")
PATENT_CATEGORY = os.path.join(PATENT2VEC_PATH, "patents.map.doctag_syn0_classes")
L2_NORMALIZED_PATENT_EMBEDDING = None
STANDARDIZED_PATENT_EMBEDDING = os.path.join(PATENT2VEC_PATH, "patents.map.doctag_syn0_standardized")


# Database options
PATENT_EMBEDDING_DATABASE_PATH = os.path.join(os.getcwd(), "output", "database")
PATENT_EMBEDDING_DATABASE = os.path.join(PATENT_EMBEDDING_DATABASE_PATH, "PatentEmbedding.db")
PATENT_EMBEDDING_TABLE = "PatentEmbedding"
PRIMARY_KEY = ("PatentName", "TEXT")
FIELDS = [("DocumentEmbedding", "TEXT"),
          ("PatentCategory", "TEXT")]
PATENT_EMBEDDING_INDEX = "PatentEmbeddingIndex"


# Clustering options
PATENT_CLUSTERING_PATH = os.path.join(os.getcwd(), "output", "clusters")
PATENT_MATRIX = os.path.join(PATENT_CLUSTERING_PATH, "PatentEmbedding.mat")
LABELS = os.path.join(PATENT_CLUSTERING_PATH, "PatentEmbedding.rlabel")
CLASSES = os.path.join(PATENT_CLUSTERING_PATH, "PatentEmbedding.rclass")
PATENT_CLUSTER = os.path.join(PATENT_CLUSTERING_PATH, "PatentCluster")
PATENT_CLUSTER_PLOT = os.path.join(PATENT_CLUSTERING_PATH, "PatentCluster.ps")
DISABLE_PATENT_CATEGORIES = False
