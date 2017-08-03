# -*- coding: utf-8 -*-

"""
@File:           Patent2VecApp.py
@Description:    This is a Patent2Vec Application.
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
import codecs
import logging
from collections import OrderedDict

from Configuration import config

from Utils.exceptions import PathNotFoundError
from Utils.exceptions import ModelNotFoundError

from Utils.database import Database

from Utils.cleanup import clean

from Preprocessing.preprocessor import PatentDocument

from Model.patent2vec import Patent2Vec
from Model.patent2vec import ConcatenatedPatent2Vec
from Model.patent2vec import AvgPatent2Vec


# Set logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s [%(levelname)s] %(message)s',)
log = logging.getLogger("Patent2Vec Application")


# Global variables
CPU_CORE = config.CPU_CORE

# SOURCE_DATASET = config.SOURCE_DATASET
SOURCE_DATASET = config.CLUSTERING_BENCHMARK_DATA
TESTING_DATA = config.TESTING_DATA

TEST_DOCUMENT = config.TEST_DOCUMENT

PRETRAINED_EMBEDDING = config.PRETRAINED_EMBEDDING

PATENT2VEC_MODEL = config.PATENT2VEC_MODEL
DOCVECS_MAP = config.DOCVECS_MAP
PATENT_EMBEDDING = config.PATENT_EMBEDDING

PATENT_EMBEDDING_DATABASE = config.PATENT_EMBEDDING_DATABASE
PATENT_EMBEDDING_TABLE = config.PATENT_EMBEDDING_TABLE
PRIMARY_KEY = config.PRIMARY_KEY
FIELDS = config.FIELDS
PATENT_EMBEDDING_INDEX = config.PATENT_EMBEDDING_INDEX


def main():
    log.info("*****Patent2Vec Application*****")

    # Preprocess patent documents
    log.info("Preprocessing patent documents")
    patents = PatentDocument(SOURCE_DATASET,
                             extension="",
                             enable_pos_tagging=True,
                             enable_lemmatization=True,
                             use_conceptualizer=True,
                             transform_conceptualizer=False,
                             enable_sampling=True,
                             train_ratio=1.0,
                             test_ratio=0.0,
                             java_options='-mx4096m')

    # Create Patent2Vec model
    models = OrderedDict()

    # PV-DM with average
    models["PV_DM_Mean"] = \
        Patent2Vec(dm=1, dm_mean=1, dm_concat=0, min_word_count=5, size=500,
                   context_window_size=8, negative=2, iter=50, workers=CPU_CORE,
                   use_less_memory=False, docvecs_mapfile=DOCVECS_MAP)
    models["PV_DM_Mean"].build(patents)
    models["PV_DM_Mean"].intersect_with_pretrained_embedding(PRETRAINED_EMBEDDING,
                                                             binary=False)
#     models["PV_DM_Mean"].load(PATENT2VEC_MODEL)

#     # PV-DM with concatenation
#     models["PV_DM_Concatenation"] = \
#         Patent2Vec(dm=1, dm_mean=0, dm_concat=1, min_word_count=5, size=500,
#                    context_window_size=8, negative=2, iter=50, workers=CPU_CORE,
#                    use_less_memory=False, docvecs_mapfile=DOCVECS_MAP)
#     models["PV_DM_Concatenation"].reuse_from(models["PV_DM_Mean"])
# #     models["PV_DM_Concatenation"].build(patents)
# #     models["PV_DM_Concatenation"].intersect_with_pretrained_embedding(PRETRAINED_EMBEDDING,
# #                                                                       binary=False)
# # #     models["PV_DM_Concatenation"].load(PATENT2VEC_MODEL)

#     # PV-DBOW
#     models["PV_DBOW"] = \
#         Patent2Vec(dm=0, dm_mean=0, dm_concat=0, min_word_count=5, size=500,
#                    context_window_size=8, negative=2, iter=50, workers=CPU_CORE,
#                    use_less_memory=False, docvecs_mapfile=DOCVECS_MAP)
#     models["PV_DBOW"].reuse_from(models["PV_DM_Mean"])
# #     models["PV_DBOW"].build(patents)
# #     models["PV_DBOW"].intersect_with_pretrained_embedding(PRETRAINED_EMBEDDING,
# #                                                           binary=False)
# # #     models["PV_DBOW"].load(PATENT2VEC_MODEL)

#     # Mixed models
#     models["DBOW + DM with average"] = ConcatenatedPatent2Vec([models["PV_DBOW"],
#                                                                models["PV_DM_Mean"]])
#     models["DBOW + DM with concatenation"] = ConcatenatedPatent2Vec([models["PV_DBOW"],
#                                                                      models["PV_DM_Concatenation"]])

    for name, model in models.items():
        # Train Patent2Vec model
        start_time = time.time()
        model.train(patents, alpha=0.1, min_alpha=0.0001, passes=10,
                    fixed_alpha=False)
        end_time = time.time()
        log.info("Total time elapsed: %r", (end_time-start_time))

        # Evaluate Patent2Vec model
        model.evaluate()

        # Save Patent2Vec model
        model.save(model=PATENT2VEC_MODEL)

        # Create a database object
        db = Database(verbose=True)

        # Connect to database
        db.connect(in_memory=True)

        # Create a new table for storing document embeddings
        db.create_table(table=PATENT_EMBEDDING_TABLE,
                        primary_column=PRIMARY_KEY,
                        other_columns=FIELDS)

        # Save document embeddings
        model.save_document_embeddings(document_embeddings=PATENT_EMBEDDING,
                                       rows=len(patents),
                                       columns=500,
                                       database=db,
                                       table_name=PATENT_EMBEDDING_TABLE,
                                       save_patent_category=True,
                                       prepend_document_category=True)

        # Test documents
        if not os.path.exists(TESTING_DATA):
            raise PathNotFoundError("Path does not exist: %s" % TESTING_DATA)

        with open(TESTING_DATA, "r") as t:
            test_documents = t.readlines()
            test_documents = map(lambda x: x.strip(), test_documents)
            test_documents = filter(None, test_documents)

        # Preprocessed test documents
        preprocessed_test_documents = patents.get_preprocessed_corpus(test_documents)

        # Predict document embeddings
        model.predict(preprocessed_test_documents,
                      alpha=0.1,
                      min_alpha=0.0001,
                      steps=50,
                      save=True,
                      database=db,
                      table_name=PATENT_EMBEDDING_TABLE,
                      save_patent_category=True,
                      prepend_document_category=True)

        # Create an index on document embedding table
        db.create_index(index=PATENT_EMBEDDING_INDEX,
                        table=PATENT_EMBEDDING_TABLE,
                        index_by_column=PRIMARY_KEY[0])

        # Close database connection
        db.close(save_to=PATENT_EMBEDDING_DATABASE)

        # Delete temporary training data
        model.clean()

    # Test document for checking the quality of Patent2Vec model
    patents.set_token_only(True)
    preprocessed_test_document = patents.get_preprocessed_document(TEST_DOCUMENT)
    patents.set_token_only(False)

    # Check quality of Patent2Vec model
    if preprocessed_test_document is not None:
        log.info("Check quality of Patent2Vec model")
        log.info("Top matches for test document: %s", TEST_DOCUMENT)

        for name, model in models.items():
            embedding = model.infer(preprocessed_test_document)

            top_matches = model.model.docvecs.most_similar(positive=[embedding],
                                                           negative=[],
                                                           topn=10)
            top_matches = map(lambda x: x[0]+"\t\t"+str(x[1]), top_matches)

            for top_match in top_matches:
                log.info(top_match)

    # Clean all un-necessary files
    clean(cleanSample=True,
          cleanModel=False,
          cleanDocvecs=True,
          cleanDatabase=False,
          cleanClusters=False,
          filter=[])


if __name__ == "__main__":
    main()
