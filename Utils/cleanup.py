# -*- coding: utf-8 -*-

"""
@File:           cleanup.py
@Description:    This is an utility for cleaning output directory.
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
import logging

from Configuration import config


# Set logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s [%(levelname)s] %(message)s',)
log = logging.getLogger("Cleanup")


# Global variables
SAMPLED_DATA_PATH = config.SAMPLED_DATA_PATH
PATENT2VEC_MODEL_PATH = config.PATENT2VEC_MODEL_PATH
PATENT2VEC_PATH = config.PATENT2VEC_PATH
PATENT_EMBEDDING_DATABASE_PATH = config.PATENT_EMBEDDING_DATABASE_PATH
PATENT_CLUSTERING_PATH = config.PATENT_CLUSTERING_PATH


def clean(cleanSample=True,
          cleanModel=True,
          cleanDocvecs=True,
          cleanDatabase=True,
          cleanClusters=True,
          filter=[]):
    """
    Clean output directory.
    """
    if cleanSample:
        _remove_documents(SAMPLED_DATA_PATH, filter)
    if cleanModel:
        _remove_documents(PATENT2VEC_MODEL_PATH, filter)
    if cleanDocvecs:
        _remove_documents(PATENT2VEC_PATH, filter)
    if cleanDatabase:
        _remove_documents(PATENT_EMBEDDING_DATABASE_PATH, filter)
    if cleanClusters:
        _remove_documents(PATENT_CLUSTERING_PATH, filter)


def _remove_documents(directory, filter=[]):
    """
    Remove documents within directory.
    """
    documents = _list_documents(directory)

    if not documents:
        return

    for document in documents:
        if document.rsplit(os.sep, 1)[1] not in filter:
            log.info("Removed: %s", document)
            os.remove(document)


def _list_documents(directory):
    """
    List all documents within directory.
    """
    if not os.path.exists(directory):
        raise PathNotFoundError("Directory does not exist: %s" % directory)

    documents = []

    for root, folders, files in os.walk(directory):
        for file in files:
            if not file.startswith('.'):
                documents.append(os.path.join(root, file))

    return documents
