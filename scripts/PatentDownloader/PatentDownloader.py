# -*- coding: utf-8 -*-

"""
@File:           PatentDownloader.py
@Description:    This is an application for downloading patents from solr index.
@Author:         Chetan Borse
@EMail:          chetanborse2106@gmail.com
@Created_on:     04/05/2017
@License         GNU General Public License
@python_version: 3.5
===============================================================================
"""


import os
import json
import logging
from urllib.request import urlopen


# Set logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s [%(levelname)s] %(message)s',)
log = logging.getLogger("PatentDownloader")


# Global variables
PATH = os.path.join(os.sep, "users", "cborse", "clustering", "data", "source", "PatentCollection")
URL = 'http://cci-text-analytics01.uncc.edu:8992/solr/collection1/select?q=application_year%3A{year}&wt=json&indent=true&rows={rows}'


def main(start_year=1900, end_year=2017, rows=1):
    year = start_year

    while year <= end_year:
        log.info("Downloading patents for year: %d", year)
        url = URL.format(year=year, rows=rows)

        with urlopen(url) as connection:
            response = eval(connection.read())
            log.info("%d patents found.", response['response']['numFound'])

            if response['response']['numFound'] > 0:
                path = os.path.join(PATH, str(year))
                if not os.path.exists(path):
                    os.makedirs(path, 0o755)

                for patent in response['response']['docs']:
                    log.info(patent["id"])
                    patent_file = os.path.join(path, patent["id"])
                    if not os.path.exists(patent_file):
                        with open(patent_file, 'w', encoding='utf-8') as p:
                            json.dump(patent, p, ensure_ascii=False)

                    #with open(patent_file, 'r') as p:
                    #    patent = json.load(p)
                    #    log.info(patent["id"])

        year += 1


if __name__ == "__main__":
    main(1900, 2017, 1000000)
