# -*- coding: utf-8 -*-

"""
@File:           20_newsgroups_extractor.py
@Description:    This is an application for preprocessing 20 newsgroup data.
@Author:         Chetan Borse
@EMail:          chetanborse2106@gmail.com
@Created_on:     04/05/2017
@License         GNU General Public License
@python_version: 3.5
===============================================================================
"""


import os
import re
import codecs


# Global variables
SOURCE_DATA = os.path.join(os.getcwd(), "data", "source", "20_newsgroups")
PREPROCESSED_DATA = os.path.join(os.getcwd(), "data", "preprocessed", "20_newsgroups")
SOURCE_ENCODING = ["utf-8", "iso8859-1", "latin1"]
LINE_FIELD_PATTERN = re.compile("\nLines:\s(\d+?)\n")
SUBJECT_FIELD_PATTERN = re.compile("\nSubject:\s(.+?)\n")


def _list_documents(source, extension=""):
    """
    """
    if not os.path.exists(source):
        raise SourceNotExistError("SourceNotExistError!!")

    documents = []
    for root, folders, files in os.walk(source):
        for file in files:
            if file.endswith(extension):
                documents.append(os.path.join(root, file))

    return documents


def main():
    documents = _list_documents(SOURCE_DATA, "")

    for document in documents:
        print("Preprocessing document: " + document)

        preprocessed_document = os.path.join(*(document.rsplit(os.sep, 2)[1:]))
        preprocessed_document = os.path.join(PREPROCESSED_DATA, preprocessed_document)

        subdirectory = preprocessed_document.rsplit(os.sep, 1)[0]
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory, mode=0o777)

        for source_encoding in SOURCE_ENCODING:
            with codecs.open(document, "r", source_encoding) as i:
                with codecs.open(preprocessed_document, "a", "utf-8") as o:
                    try:
                        content = i.read()
                    except UnicodeDecodeError as e:
                        continue

                    line_field_matches = re.search(LINE_FIELD_PATTERN, content)
                    if line_field_matches:
                        total_lines = int(line_field_matches.group(1))

                    subject_field_matches = re.search(SUBJECT_FIELD_PATTERN,
                                                      content)
                    if subject_field_matches:
                        subject = subject_field_matches.group(1)

                    content = content.strip()
                    content = content.split("\n")
                    article = content[len(content)-total_lines:]
                    article = "\n".join(article)

                    o.write("Subject: " + subject + "\n")
                    o.write(article)

                    break


if __name__ == "__main__":
    main()
