# -*- coding: utf-8 -*-

"""
@File:           ClusteringApp.py
@Description:    This is a Clustering Application.
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

from Configuration import config

from Utils.exceptions import PathNotFoundError
from Utils.exceptions import ModelNotFoundError

from Utils.database import Database

from Utils.cleanup import clean

from Model.clustering import Clustering
from Model.clustering import ClusteringError


# Set logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s [%(levelname)s] %(message)s',)
log = logging.getLogger("Clustering Application")


# Global variables
PATENT_EMBEDDING_DATABASE = config.PATENT_EMBEDDING_DATABASE
PRIMARY_KEY = config.PRIMARY_KEY

PATENT_CLUSTERING_PATH = config.PATENT_CLUSTERING_PATH
PATENT_MATRIX = config.PATENT_MATRIX
LABELS = config.LABELS
CLASSES = config.CLASSES
PATENT_CLUSTER = config.PATENT_CLUSTER
PATENT_CLUSTER_PLOT = config.PATENT_CLUSTER_PLOT


def main():
    log.info("*****Clustering Application*****")

    # Create a model for clustering patents
    model = Clustering(method="rbr",
                       criterion="i2",
                       similarity="cos",
                       cluster_choice="best",
                       rowmodel="none",
                       colmodel="none",
                       trials=10,
                       showfeatures=False,
                       showsummaries=True,
                       summary_method="cliques",
                       showtree=False,
                       zscores=False,
                       plotclusters=True,
                       plotformat="ps")

    # Create an object of 'Database'
    db = Database(verbose=True)

    # Connect to SQLite database
    db.connect(in_memory=True, load_from=PATENT_EMBEDDING_DATABASE)

    # Dummy document collection
    documents = []
    for root, folders, files in os.walk(config.CLUSTERING_BENCHMARK_DATA):
        for file in files:
            if not file.startswith('.'):
                if file.endswith(""):
                    document_name = file
                    document_category = root.rsplit(os.sep, 1)[1]
                    document_label = document_category + "." + document_name
                    documents.append(document_label)

    # Generate matrix of document embeddings
    model.patent2mat(documents,
                     rows=len(documents),
                     columns=500,
                     database=db,
                     search_on=PRIMARY_KEY,
                     matrix=PATENT_MATRIX,
                     labels=LABELS,
                     classes=CLASSES,
                     path=PATENT_CLUSTERING_PATH)

    # Close connection to SQLite database
    db.close()

    # Cluster documents
    model.train(matrix=PATENT_MATRIX,
                labels=LABELS,
                classes=CLASSES,
                use_patent_classes=True,
                k=20,
                iterations=20,
                patent_clusters=PATENT_CLUSTER,
                plot=PATENT_CLUSTER_PLOT,
                path=PATENT_CLUSTERING_PATH)

    # Clean all un-necessary files
    clean(cleanSample=True,
          cleanModel=False,
          cleanDocvecs=True,
          cleanDatabase=False,
          cleanClusters=True,
          filter=["PatentCluster", "PatentCluster.ps", "PatentEmbedding.rclass"])


if __name__ == "__main__":
    main()
