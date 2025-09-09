"""
check_train.py
This file contains script to run all given models and then plot results of their learning.
"""

import logging

from src.helpers.config.const import *
from src.helpers.config.dir_config import *
from src.helpers.logs.log_parser import LogParser

logging.basicConfig(level=logging.INFO)

DATASETS_CITATION_NETWORKS = ["cora", "citeseer"]
DATASETS_SOCIAL_NETWORKS = ["BlogCatalog", "Flickr"]
DATASETS_UNDER_SAME_HASHTAG = ["weibo"]
DATASETS_WORK_COLLABORATION = ["tolokers"]
DATASETS_CO_PURCHASE = ["Disney", "books"]
DATASETS_USER_SUBREDDIT = ["Reddit"]

def load_all_results():
    parser_1 = LogParser(log_dir=RESULTS_DIR_ANOMALYDAE)
    parser_1.parse_logs()
    parser_2 = LogParser(log_dir=RESULTS_DIR_COLA)
    parser_2.parse_logs()
    parser_3 = LogParser(log_dir=RESULTS_DIR_OCGNN)
    parser_3.parse_logs()

    all_results = parser_1.results + parser_2.results + parser_3.results
    all_results = [
        result
        for result in all_results
        if (result[DICT_EPOCH] == 100) and
           (result[DICT_FEATURE_LABEL] != "Attr + Alpha1") and
           (result[DICT_FEATURE_LABEL] != "Attr + Alpha2")
    ]
    all_datasets = set(result[DICT_DATASET] for result in all_results)

    return all_results, all_datasets

all_results, all_datasets = load_all_results()
all_models = ["AnomalyDAE", "CoLA", "OCGNN"]

def get_results_of_some_datasets(some_datasets = all_datasets, results = all_results):
    return [
        result
        for result in results
        if result[DICT_DATASET] in some_datasets
    ]

def get_results_of_some_models(some_models, results = all_results):
    return [
        result
        for result in results
        if result[DICT_MODEL] in some_models
    ]

def find_best_model_in_some_results(results = all_results):
    set_auc_roc = find_best_model_through_auc_roc_in_some_results(results)
    set_recall = find_best_model_through_recall_in_some_results(results)
    set_precision = find_best_model_through_precision_in_some_results(results)

    intersection = set_auc_roc & set_recall & set_precision

    if len(intersection) == 0:
        print("     - Through all 3 metrics: Not found")
        print(f"        Through AUC-ROC {set_auc_roc}")
        print(f"        Through Precision {set_precision}")
        print(f"        Through Recall {set_recall}")
    else:
        print(f"     - Through all 3 metrics: Found {intersection}")

def find_best_enrichment_in_some_results(results = all_results):
    set_auc_roc = find_best_enrichment_through_auc_roc_in_some_results(results)
    set_recall = find_best_enrichment_through_recall_in_some_results(results)
    set_precision = find_best_enrichment_through_precision_in_some_results(results)

    intersection = set_auc_roc & set_recall & set_precision

    if len(intersection) == 0:
        print("     - Through all 3 metrics: Not found")
        print(f"        Through AUC-ROC {set_auc_roc}")
        print(f"        Through Precision {set_precision}")
        print(f"        Through Recall {set_recall}")
    else:
        print(f"     - Through all 3 metrics: Found {intersection}")


def find_best_model_through_some_metric_in_some_results(some_metric, results = all_results):
    models = []
    if (some_metric != DICT_AUC_ROC):
        max = -1
    else:
        max = 0

    for result in results:
        if (some_metric == DICT_AUC_ROC):
            metric_result = abs(0.5 - result.get(some_metric))
        else:
            metric_result = result.get(some_metric)
        if metric_result > max:
            max = metric_result

    for result in results:
        if (some_metric == DICT_AUC_ROC):
            metric_result = abs(0.5 - result.get(some_metric))
        else:
            metric_result = result.get(some_metric)
        if metric_result == max:
            models.append(result.get(DICT_MODEL))

    return set(models)

def find_best_enrichment_through_some_metric_in_some_results(some_metric, results = all_results):
    enrichments = []
    if (some_metric != DICT_AUC_ROC):
        max = -1
    else:
        max = 0

    for result in results:
        if (some_metric == DICT_AUC_ROC):
            metric_result = abs(0.5 - result.get(some_metric))
        else:
            metric_result = result.get(some_metric)
        if metric_result > max:
            max = metric_result

    for result in results:
        if (some_metric == DICT_AUC_ROC):
            metric_result = abs(0.5 - result.get(some_metric))
        else:
            metric_result = result.get(some_metric)
        if metric_result == max:
            enrichments.append(result.get(DICT_FEATURE_LABEL))

    return set(enrichments)

def find_best_model_through_precision_in_some_results(results = all_results):
    return find_best_model_through_some_metric_in_some_results(DICT_PRECISION, results)

def find_best_model_through_auc_roc_in_some_results(results = all_results):
    return find_best_model_through_some_metric_in_some_results(DICT_AUC_ROC, results)

def find_best_model_through_recall_in_some_results(results = all_results):
    return find_best_model_through_some_metric_in_some_results(DICT_RECALL, results)

def find_best_enrichment_through_precision_in_some_results(results = all_results):
    return find_best_enrichment_through_some_metric_in_some_results(DICT_PRECISION, results)

def find_best_enrichment_through_auc_roc_in_some_results(results = all_results):
    return find_best_enrichment_through_some_metric_in_some_results(DICT_AUC_ROC, results)

def find_best_enrichment_through_recall_in_some_results(results = all_results):
    return find_best_enrichment_through_some_metric_in_some_results(DICT_RECALL, results)

def find_best_model_through_for_some_datasets(datasets = all_datasets):
    results = get_results_of_some_datasets(datasets)
    find_best_model_in_some_results(results)

def find_best_enrichment_through_for_some_datasets(datasets = all_datasets):
    results = get_results_of_some_datasets(datasets)
    find_best_enrichment_in_some_results(results)




def find_best_model_through_for_citation_networks():
    print("Best for citation networks")
    find_best_model_through_for_some_datasets(DATASETS_CITATION_NETWORKS)

def find_best_model_through_for_social_networks():
    print("Best for social networks")
    find_best_model_through_for_some_datasets(DATASETS_SOCIAL_NETWORKS)

def find_best_model_through_for_under_same_hashtag():
    print("Best for under same hashtag")
    find_best_model_through_for_some_datasets(DATASETS_UNDER_SAME_HASHTAG)

def find_best_model_through_for_work_collaboration():
    print("Best for under work collaboration")
    find_best_model_through_for_some_datasets(DATASETS_WORK_COLLABORATION)

def find_best_model_through_for_co_purchase():
    print("Best for under co purchase")
    find_best_enrichment_through_for_some_datasets(DATASETS_CO_PURCHASE)

def find_best_model_through_for_user_subreddit():
    print("Best for under user subreddit")
    find_best_model_through_for_some_datasets(DATASETS_USER_SUBREDDIT)

def find_best_enrichment_through_for_citation_networks():
    print("Best for citation networks")
    find_best_enrichment_through_for_some_datasets(DATASETS_CITATION_NETWORKS)

def find_best_enrichment_through_for_social_networks():
    print("Best for social networks")
    find_best_enrichment_through_for_some_datasets(DATASETS_SOCIAL_NETWORKS)

def find_best_enrichment_through_for_under_same_hashtag():
    print("Best for under same hashtag")
    find_best_enrichment_through_for_some_datasets(DATASETS_UNDER_SAME_HASHTAG)

def find_best_enrichment_through_for_work_collaboration():
    print("Best for under work collaboration")
    find_best_enrichment_through_for_some_datasets(DATASETS_WORK_COLLABORATION)

def find_best_enrichment_through_for_co_purchase():
    print("Best for under co purchase")
    find_best_enrichment_through_for_some_datasets(DATASETS_CO_PURCHASE)

def find_best_enrichment_through_for_user_subreddit():
    print("Best for under user subreddit")
    find_best_enrichment_through_for_some_datasets(DATASETS_USER_SUBREDDIT)

def find_best_models_for_domains():
    find_best_model_through_for_co_purchase()
    find_best_model_through_for_under_same_hashtag()
    find_best_model_through_for_citation_networks()
    find_best_model_through_for_social_networks()
    find_best_model_through_for_work_collaboration()
    find_best_model_through_for_user_subreddit()
    print("\n\n")

def find_best_enrichments_for_domains():
    find_best_enrichment_through_for_co_purchase()
    find_best_enrichment_through_for_under_same_hashtag()
    find_best_enrichment_through_for_citation_networks()
    find_best_enrichment_through_for_social_networks()
    find_best_enrichment_through_for_work_collaboration()
    find_best_enrichment_through_for_user_subreddit()
    print("\n\n")



def main():
    find_best_enrichments_for_domains()


if __name__ == "__main__":
    main()
