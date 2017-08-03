# -*- coding: utf-8 -*-

"""
@File:           evaluator.py
@Description:    This is an application for evaluating clustering performance
                 on 20 newsgroup data.
@Author:         Chetan Borse
@EMail:          chetanborse2106@gmail.com
@Created_on:     04/05/2017
@python_version: 3.5
===============================================================================
"""


import os
import logging


# Set logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s [%(levelname)s] %(message)s',)
log = logging.getLogger("ClusteringEvaluation")


# Global variables
TRUE_LABEL = os.path.join(os.getcwd(),
                          "output",
                          "clusters",
                          "PatentEmbedding.rclass")
PREDICTED_LABEL = os.path.join(os.getcwd(),
                               "output",
                               "clusters",
                               "PatentCluster")

SUBCLUSTER_MAP = {"0": "rec.sport.hockey",
                  "1": "rec.sport.baseball",
                  "2": "comp.sys.ibm.pc.hardware",
                  "3": "comp.windows.x",
                  "4": "misc.forsale",
                  "5": "sci.crypt",
                  "6": "sci.crypt",
                  "7": "talk.politics.guns",
                  "8": "comp.os.ms-windows.misc",
                  "9": "comp.sys.mac.hardware",
                  "10": "rec.autos",
                  "11": "rec.motorcycles",
                  "12": "soc.religion.christian",
                  "13": "sci.med",
                  "14": "talk.politics.mideast",
                  "15": "comp.graphics",
                  "16": "alt.atheism",
                  "17": "sci.electronics",
                  "18": "sci.space",
                  "19": "talk.politics.misc"}

SUPERCLUSTER_MAP = {"comp.sys.ibm.pc.hardware": 0,
                    "comp.windows.x": 0,
                    "comp.os.ms-windows.misc": 0,
                    "comp.sys.mac.hardware": 0,
                    "comp.graphics": 0,
                    "rec.sport.hockey": 1,
                    "rec.sport.baseball": 1,
                    "rec.autos": 1,
                    "rec.motorcycles": 1,
                    "sci.crypt": 2,
                    "sci.med": 2,
                    "sci.space": 2,
                    "sci.electronics": 2,
                    "misc.forsale": 3,
                    "talk.politics.guns": 4,
                    "talk.politics.mideast": 4,
                    "talk.politics.misc": 4,
                    "soc.religion.christian": 5,
                    "alt.atheism": 5,
                    "talk.religion.misc": 5}


def main():
    log.info("Performing clustering evaluation")

    # Statistics for individual cluster
    subCluster = {}
    superCluster = {}

    # Initialize sub-clusters' and super-clusters' statistics
    # as a triplet [<True Predictions> <Total Predictions> <Purity>]
    for i in range(20):
        subCluster[str(i)] = [0, 0, 0]
        superCluster[str(i)] = [0, 0, 0]

    with open(TRUE_LABEL, "r") as t, open(PREDICTED_LABEL, "r") as p:
        trueLabels = t.readlines()
        trueLabels = map(lambda x: x.strip(), trueLabels)

        predictedLabels = p.readlines()
        predictedLabels = list(map(lambda x: x.strip(), predictedLabels))

        # Calculate 'True Predictions' and 'Total Predictions'
        # for every cluster
        for i, trueLabel in enumerate(trueLabels):
            predictedLabel = predictedLabels[i]

            if trueLabel == SUBCLUSTER_MAP[predictedLabel]:
                subCluster[predictedLabel][0] += 1
            subCluster[predictedLabel][1] += 1

            if (SUPERCLUSTER_MAP[trueLabel] == 
                    SUPERCLUSTER_MAP[SUBCLUSTER_MAP[predictedLabel]]):
                superCluster[predictedLabel][0] += 1
            superCluster[predictedLabel][1] += 1

    # Calculate 'Purity' for every cluster
    for i in range(20):
        subCluster[str(i)][2] = float(subCluster[str(i)][0]) / subCluster[str(i)][1]
        superCluster[str(i)][2] = float(superCluster[str(i)][0]) / superCluster[str(i)][1]

    log.info("Sub-cluster evaluation: ")
    log.info("True Predictions\tTotal Predictions\tPurity")
    for i in range(20):
        log.info("%d\t\t\t%d\t\t\t%f",
                 subCluster[str(i)][0],
                 subCluster[str(i)][1],
                 subCluster[str(i)][2])

    log.info("Super-cluster evaluation: ")
    log.info("True Predictions\tTotal Predictions\tPurity")
    for i in range(20):
        log.info("%d\t\t\t%d\t\t\t%f",
                 superCluster[str(i)][0],
                 superCluster[str(i)][1],
                 superCluster[str(i)][2])

    # Overall statistics for sub-clusters and super-clusters
    overallStatistics = {}
    overallStatistics["SubCluster"] = [0, 0, 0]
    overallStatistics["SuperCluster"] = [0, 0, 0]

    # Overall statistics for sub-clusters
    truePredictions = 0
    totalPredictions = 0
    for cluster, stat in subCluster.items():
        truePredictions += stat[0]
        totalPredictions += stat[1]
    overallStatistics["SubCluster"][0] = truePredictions
    overallStatistics["SubCluster"][1] = totalPredictions
    overallStatistics["SubCluster"][2] = float(truePredictions) / totalPredictions

    # Overall statistics for super-clusters
    truePredictions = 0
    totalPredictions = 0
    for cluster, stat in superCluster.items():
        truePredictions += stat[0]
        totalPredictions += stat[1]
    overallStatistics["SuperCluster"][0] = truePredictions
    overallStatistics["SuperCluster"][1] = totalPredictions
    overallStatistics["SuperCluster"][2] = float(truePredictions) / totalPredictions

    log.info("Overall statistics for sub-clusters: ")
    log.info("True Predictions\tTotal Predictions\tPurity")
    log.info("%d\t\t%d\t\t\t%f",
             overallStatistics["SubCluster"][0],
             overallStatistics["SubCluster"][1],
             overallStatistics["SubCluster"][2])

    log.info("Overall statistics for super-clusters: ")
    log.info("True Predictions\tTotal Predictions\tPurity")
    log.info("%d\t\t%d\t\t\t%f",
             overallStatistics["SuperCluster"][0],
             overallStatistics["SuperCluster"][1],
             overallStatistics["SuperCluster"][2])

    # Combined overall statistics for both sub-clusters and super-clusters
    truePredictions = 0
    totalPredictions = 0
    for clusterType, stat in overallStatistics.items():
        truePredictions += stat[0]
        totalPredictions += stat[1]
    combinedOverallPurity = float(truePredictions) / totalPredictions

    log.info("Combined overall statistics for both sub-clusters and super-clusters: ")
    log.info("True Predictions\tTotal Predictions\tPurity")
    log.info("%d\t\t%d\t\t\t%f",
             truePredictions,
             totalPredictions,
             combinedOverallPurity)


if __name__ == "__main__":
    main()
