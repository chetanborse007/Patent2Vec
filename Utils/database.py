# -*- coding: utf-8 -*-

"""
@File:           database.py
@Description:    This is a module for different database operations and
                 to provide a fast lookup.

                 This application,
                     1. Open/Close a SQLite database connection
                     2. Create a new SQLite database
                     3. Create a new SQLite table
                     4. Insert records into SQLite table
                     5. Create a new index on SQLite table for efficient lookup
                     6. Drop an index
                     7. Retrieve records from SQLite table
                        for a provided condition
                     8. Find out the total number of records in the database
                     9. Find out the table schema
                     10. Save/Load database to/from disk
                     11. Perform fast lookup on database
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
import math
import time
import logging
from functools import partial
from multiprocessing import Pool
from multiprocessing import Lock

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import numpy as np

import sqlite3

from Configuration import config


# Set logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s [%(levelname)s] %(message)s',)
log = logging.getLogger("Database")


# Global variables
PATENT_EMBEDDING_DATABASE = config.PATENT_EMBEDDING_DATABASE
PATENT_EMBEDDING_TABLE = config.PATENT_EMBEDDING_TABLE
PRIMARY_KEY = config.PRIMARY_KEY
FIELDS = config.FIELDS
PATENT_EMBEDDING_INDEX = config.PATENT_EMBEDDING_INDEX

PATENT_CLUSTERING_PATH = config.PATENT_CLUSTERING_PATH
PATENT_MATRIX = config.PATENT_MATRIX
LABELS = config.LABELS
CLASSES = config.CLASSES
DISABLE_PATENT_CATEGORIES = config.DISABLE_PATENT_CATEGORIES


# Lock for synchronized access
LOCK = Lock()


class DatabaseError(Exception):
    pass


class FileHandlerError(Exception):
    pass


class Database(object):
    """
    This is a class for different database operations.

    This class,
        1. Open/Close a SQLite database connection
        2. Create a new SQLite database
        3. Create a new SQLite table
        4. Insert records into SQLite table
        5. Create a new index on SQLite table for efficient lookup
        6. Drop an index
        7. Retrieve records from SQLite table
           for a provided condition
        8. Find out the total number of records in the database
        9. Find out the table schema
        10. Save/Load database to/from disk
    """

    def __init__(self, verbose=False):
        self.connection = None
        self.cursor = None
        self.verbose = verbose

    def connect(self,
                database=PATENT_EMBEDDING_DATABASE,
                in_memory=True,
                load_from=None):
        """
        Connect to a SQLite database.
        """
        try:
            if in_memory:
                self.connection = sqlite3.connect(':memory:')
            else:
                self.connection = sqlite3.connect(database)

            self.cursor = self.connection.cursor()

            if load_from is not None:
                with open(load_from, "r") as f:
                    self.cursor.executescript(f.read())
                    self.connection.commit()
        except IOError as e:
            raise DatabaseError("Database application failed with: %s" % e)
        except sqlite3.OperationalError as e:
            raise DatabaseError("Database application failed with: %s" % e)
        except sqlite3.Error as e:
            raise DatabaseError("Database application failed with: %s" % e)
        except Exception as e:
            raise DatabaseError("Database application failed with: %s" % e)

    def create_table(self,
                     table=PATENT_EMBEDDING_TABLE,
                     primary_column=PRIMARY_KEY,
                     other_columns=FIELDS):
        """
        Create a new SQLite table.
        """
        try:
            self.cursor.execute('CREATE TABLE {tn} ({f} {t} NOT NULL PRIMARY KEY)' \
                                .format(tn=table,
                                        f=primary_column[0], t=primary_column[1]))

            for column, type in other_columns:
                self.cursor.execute("ALTER TABLE {tn} ADD COLUMN '{f}' {t}" \
                                    .format(tn=table, f=column, t=type))
        except sqlite3.OperationalError as e:
            raise DatabaseError("Database application failed with: %s" % e)
        except sqlite3.Error as e:
            raise DatabaseError("Database application failed with: %s" % e)
        except Exception as e:
            raise DatabaseError("Database application failed with: %s" % e)

        self.connection.commit()

    def insert(self,
               table=PATENT_EMBEDDING_TABLE,
               record=[("PatentName", None),
                       ("DocumentEmbedding", None),
                       ("PatentCategory", "UNKNOWN")]):
        """
        Insert records into SQLite table.
        """
        query = "INSERT OR IGNORE INTO {tn} ({f}) VALUES ({v})"
        columns = map(lambda x: x[0], record)
        values = map(lambda x: '\''+str(x[1])+'\'', record)
        columns = ", ".join(columns)
        values = ", ".join(values)
        query = query.format(tn=table, f=columns, v=values)

        self._execute_query(query)

        self.connection.commit()

    def create_index(self,
                     index=PATENT_EMBEDDING_INDEX,
                     table=PATENT_EMBEDDING_TABLE,
                     index_by_column=PRIMARY_KEY[0]):
        """
        Create a new index on SQLite table for efficient lookup.
        """
        query = 'CREATE UNIQUE INDEX {i} ON {tn} ({f})'.format(i=index,
                                                               tn=table,
                                                               f=index_by_column)

        self._execute_query(query)

        self.connection.commit()

    def drop_index(self, index):
        """
        Drop an index from a SQLite table.
        """
        query = 'DROP INDEX {i}'.format(i=index)

        self._execute_query(query)

        self.connection.commit()

    def get(self,
            table=PATENT_EMBEDDING_TABLE,
            index=PATENT_EMBEDDING_INDEX,
            required_columns=["*"],
            condition=""):
        """
        Retrieve records from SQLite table for a provided condition.
        """
        query = "SELECT {f} FROM {tn} INDEXED BY {i} WHERE {c}"
        query = query.format(f=", ".join(required_columns),
                             tn=table,
                             i=index,
                             c=condition)

        self._execute_query(query)

        records = []
        while True:
            partial_records = self.cursor.fetchmany(True)

            if not partial_records:
                break

            for record in partial_records:
                if self.verbose:
                    log.debug("%r", record)
                records.append(record)

        return records

    def get_total_records(self, table):
        """
        Returns the total number of records in the database.
        """
        query = 'SELECT COUNT(*) FROM {}'.format(table)

        self._execute_query(query)

        total_records = self.cursor.fetchall()

        if self.verbose:
            log.info('Total records: {}'.format(total_records[0][0]))

        return total_records[0][0]

    def get_table_schema(self, table):
        """ 
        Returns the table schema.
        """
        query = 'PRAGMA TABLE_INFO({})'.format(table)

        self._execute_query(query)

        table_schema = self.cursor.fetchall()

        if self.verbose:
            log.info("ID, Name, Type, Not_Null, Default_Value, Primary_Key")
            for column in table_schema:
                log.info(column)

        return table_schema

    def close(self, save_to=None):
        """
        Close connection to the database.
        """
        try:
            if self.connection:
                if save_to is not None:
                    if not os.path.exists(save_to.rsplit(os.sep, 1)[0]):
                        raise PathNotFoundError("Path does not exist: %s"
                                                % save_to.rsplit(os.sep, 1)[0])

                    with open(save_to, 'w') as f:
                        for line in self.connection.iterdump():
                            f.write('%s\n' % line)

                self.connection.close()
        except IOError as e:
            raise DatabaseError("Database application failed with: %s" % e)
        except sqlite3.OperationalError as e:
            raise DatabaseError("Database application failed with: %s" % e)
        except sqlite3.Error as e:
            raise DatabaseError("Database application failed with: %s" % e)
        except Exception as e:
            raise DatabaseError("Database application failed with: %s" % e)

    def _execute_query(self, query):
        """
        Execute SQLite query.
        """
        try:
            with LOCK:
                self.cursor.execute(query)
        except sqlite3.ProgrammingError as e:
            raise DatabaseError("Database application failed with: %s" % e)
        except sqlite3.IntegrityError as e:
            raise DatabaseError("Database application failed with: %s" % e)
        except sqlite3.OperationalError as e:
            raise DatabaseError("Database application failed with: %s" % e)
        except sqlite3.Error as e:
            raise DatabaseError("Database application failed with: %s" % e)
        except Exception as e:
            raise DatabaseError("Database application failed with: %s" % e)


class FileHandler(object):
    """
    Class for saving records retrieved from the database
    in a synchronized fashion.
    """

    @staticmethod
    def write(records, filename, mode):
        """
        Save records retrieved from the database in a synchronized fashion
        using mutex lock on shared file resource.
        """
        with LOCK:
            try:
                with open(filename, mode) as f:
                    f.write(records)
                    f.flush()
                    os.fsync(f.fileno())
            except IOError as e:
                raise FileHandlerError("FileHandler failed: %s" % filename)


def Lookup(database,
           table=PATENT_EMBEDDING_TABLE,
           index=PATENT_EMBEDDING_INDEX,
           required_columns=["*"],
           search_on=PRIMARY_KEY[0],
           save=True,
           patents=list()):
    """
    Perform lookup on database.
    """
    condition = "{s} IN ({i})"
#     condition = "{s} IN ({i}) ORDER BY FIELD ({o})"
    patents = map(lambda x: '\''+str(x)+'\'', patents)
    patents = ",".join(patents)
    condition = condition.format(s=search_on, i=patents)
#     condition = condition.format(s=search_on, i=patents, o=patents)

    records = database.get(table, index, required_columns, condition)

    if save:
        SaveRecords(records)

    return records


def FastLookup(database,
               table=PATENT_EMBEDDING_TABLE,
               index=PATENT_EMBEDDING_INDEX,
               required_columns=["*"],
               search_on=PRIMARY_KEY[0],
               patents=list(),
               total_processes=1,
               save=True,
               path=os.getcwd(),
               return_from=False):
    """
    Perform fast lookup on database.
    """
    chunk_size = math.ceil(float(len(patents)) / total_processes)
    if chunk_size == 0:
        chunk_size = 1

    with Pool(processes=total_processes) as pool:
        f = partial(Lookup,
                    database, table, index, required_columns, search_on, save)
        result = pool.map(f, GetChunks(patents, size=chunk_size))

    if return_from:
        return result


def GetChunks(data, size=None):
    """
    Get chunks of the data.
    """
    if size == None:
        size = len(data)

    start = 0
    end = size
    chunks = []

    while start < len(data):
        chunks.append(data[start:end])

        start = end
        end += size
        if end > len(data):
            end = len(data)

    return chunks


def SaveRecords(records):
    """
    Save records retrieved from the database.
    """
    patent_names = map(lambda x: x[0], records)
    patent_names = filter(None, patent_names)
    patent_names = "\n".join(patent_names)

    document_embeddings = map(lambda x: x[1], records)
    document_embeddings = filter(None, document_embeddings)
    document_embeddings = "\n".join(document_embeddings)

    if not DISABLE_PATENT_CATEGORIES:
        patent_categories = map(lambda x: x[2], records)
        patent_categories = filter(None, patent_categories)
        patent_categories = "\n".join(patent_categories)

    if os.path.exists(PATENT_CLUSTERING_PATH):
        if patent_names:
            FileHandler.write(patent_names+"\n", LABELS, "a")

        if document_embeddings:
            FileHandler.write(document_embeddings.encode()+b"\n",
                              PATENT_MATRIX,
                              "ab")

        if (not DISABLE_PATENT_CATEGORIES and patent_categories):
            FileHandler.write(patent_categories+"\n", CLASSES, "a")


if __name__ == '__main__':
    # Database: write operations
    db = Database(verbose=True)

    db.connect(in_memory=True)

    db.create_table(table=PATENT_EMBEDDING_TABLE,
                    primary_column=PRIMARY_KEY,
                    other_columns=FIELDS)

    total_records = 1000
    dimension = 500
    for i in range(total_records):
        default_embedding = np.zeros((dimension,), dtype=np.float32)
        document_embedding = " ".join(map(str, default_embedding))

        record = [("PatentName", str(i)),
                  ("DocumentEmbedding", document_embedding),
                  ("PatentCategory", "UNKNOWN")]

        db.insert(table=PATENT_EMBEDDING_TABLE, record=record)

    db.create_index(index=PATENT_EMBEDDING_INDEX,
                    table=PATENT_EMBEDDING_TABLE,
                    index_by_column=PRIMARY_KEY[0])

    db.get_total_records(PATENT_EMBEDDING_TABLE)
    db.get_table_schema(PATENT_EMBEDDING_TABLE)

    db.close(save_to=PATENT_EMBEDDING_DATABASE)

    # Database: read operations
    db = Database(verbose=True)

    db.connect(in_memory=True, load_from=PATENT_EMBEDDING_DATABASE)

    total_patents = 50
    patents = [str(i+5) for i in range(total_patents)]
    dimension = 500
    try:
        FileHandler.write((b"%d %d\n" % (total_patents, dimension)),
                          PATENT_MATRIX,
                          "ab")
    except IOError as e:
        raise FileHandlerError()

    start_time = time.time()
    Lookup(db,
           table=PATENT_EMBEDDING_TABLE,
           index=PATENT_EMBEDDING_INDEX,
           search_on=PRIMARY_KEY[0],
           save=True,
           patents=patents)
#     FastLookup(db, patents=patents, total_processes=4, save=True)
    end_time = time.time()
    print(end_time-start_time)

    db.close()
