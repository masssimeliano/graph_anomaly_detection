"""
check_hypotheses.py
This file contains script to calculate and answer the common research questions.
"""

import logging
from collections import Counter

from src.helpers.config.const import *
from src.helpers.config.dir_config import *
from src.helpers.config.training_config import EPOCH_TO_LEARN
from src.helpers.logs.log_parser import LogParser

logging.basicConfig(level=logging.INFO)

DATASETS_CITATION_NETWORKS = ["cora", "citeseer"]
DATASETS_SOCIAL_NETWORKS = ["BlogCatalog", "Flickr"]
DATASETS_UNDER_SAME_HASHTAG = ["weibo"]
DATASETS_WORK_COLLABORATION = ["tolokers"]
DATASETS_CO_PURCHASE = ["Disney", "book"]
DATASETS_USER_SUBREDDIT = ["Reddit"]

ONLY_ATTR_ANOMALY_DATASETS = ["tolokers"]
STR_OR_ATTR_ANOMALY_DATASET = ["BlogCatalog", "citeseer", "cora", "Flickr"]
STR_AND_ATTR_ANOMALY_DATASETS = ["Disney", "book", "computers", "cs", "photo", "weibo", "Reddit"]

PRECOMPUTED_ENRICHMENTS = [
    FEATURE_LABEL_STR2,
    FEATURE_LABEL_STR3,
    FEATURE_LABEL_STR
]
LEARNED_ENRICHMENTS = [
    FEATURE_LABEL_EMD1,
    FEATURE_LABEL_EMD2,
    FEATURE_LABEL_ERROR1,
    FEATURE_LABEL_ERROR2
]


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
        if (result[DICT_EPOCH] == EPOCH_TO_LEARN) and
           (result[DICT_FEATURE_LABEL] != FEATURE_LABEL_ALPHA1) and
           (result[DICT_FEATURE_LABEL] != FEATURE_LABEL_ALPHA2)
    ]
    all_datasets = set(result[DICT_DATASET] for result in all_results)

    return all_results, all_datasets


all_results, all_datasets = load_all_results()
all_models = ["AnomalyDAE", "CoLA", "OCGNN"]
all_enrichments = [
    FEATURE_LABEL_STR2,
    FEATURE_LABEL_STR3,
    FEATURE_LABEL_STR,
    FEATURE_LABEL_EMD1,
    FEATURE_LABEL_EMD2,
    FEATURE_LABEL_ERROR1,
    FEATURE_LABEL_ERROR2,
    FEATURE_LABEL_STANDARD,
]


def get_results_of_some_datasets(datasets, results):
    return [
        result
        for result in results
        if result[DICT_DATASET] in datasets
    ]


def get_results_of_some_models(models, results):
    return [
        result
        for result in results
        if result[DICT_MODEL] in models
    ]


def get_results_of_some_enrichments(enrichments, results):
    return [
        result
        for result in results
        if result[DICT_FEATURE_LABEL] in enrichments
    ]


def find_enrichment_through_auc_roc_for_domain(datasets):
    results = get_results_of_some_datasets(datasets, all_results)

    counts = Counter(all_enrichments)
    counts = dict(counts)
    for k, v in counts.items():
        counts[k] = 0

    for dataset in datasets:
        for model in all_models:
            max_current = 0

            for enrichment in all_enrichments:
                result_current = [
                    result
                    for result in results
                    if (result[DICT_DATASET] == dataset) and
                       (result[DICT_MODEL] == model) and
                       (result[DICT_FEATURE_LABEL] == enrichment)
                ][0]
                result_current_auc_roc = abs(0.5 - result_current[DICT_AUC_ROC])
                if result_current_auc_roc > max_current:
                    max_current = result_current_auc_roc

            for enrichment in all_enrichments:
                result_current = [
                    result
                    for result in results
                    if (result[DICT_DATASET] == dataset) and
                       (result[DICT_MODEL] == model) and
                       (result[DICT_FEATURE_LABEL] == enrichment)
                ][0]
                result_current_auc_roc = abs(0.5 - result_current[DICT_AUC_ROC])
                if result_current_auc_roc == max_current:
                    counts[enrichment] += 1

    max_value = max(counts.values())
    max_items = {k: v for k, v in counts.items() if v == max_value}

    print(f"    - By AUC-ROC: {max_items}")
    return set(max_items.keys())


def find_enrichment_through_precision_for_domain(datasets):
    results = get_results_of_some_datasets(datasets, all_results)

    counts = Counter(all_enrichments)
    counts = dict(counts)
    for k, v in counts.items():
        counts[k] = 0

    for dataset in datasets:
        for model in all_models:
            max_current = -1

            for enrichment in all_enrichments:
                result_current = [
                    result
                    for result in results
                    if (result[DICT_DATASET] == dataset) and
                       (result[DICT_MODEL] == model) and
                       (result[DICT_FEATURE_LABEL] == enrichment)
                ][0]
                result_current_precision = result_current[DICT_PRECISION]
                if result_current_precision > max_current:
                    max_current = result_current_precision

            for enrichment in all_enrichments:
                result_current = [
                    result
                    for result in results
                    if (result[DICT_DATASET] == dataset) and
                       (result[DICT_MODEL] == model) and
                       (result[DICT_FEATURE_LABEL] == enrichment)
                ][0]
                result_current_precision = result_current[DICT_PRECISION]
                if result_current_precision == max_current:
                    counts[enrichment] += 1

    max_value = max(counts.values())
    max_items = {k: v for k, v in counts.items() if v == max_value}

    print(f"    - By Precision: {max_items}")
    return set(max_items.keys())


def find_enrichment_through_recall_for_domain(datasets):
    results = get_results_of_some_datasets(datasets, all_results)

    counts = Counter(all_enrichments)
    counts = dict(counts)
    for k, v in counts.items():
        counts[k] = 0

    for dataset in datasets:
        for enrichment in all_enrichments:
            max_current = -1

            for model in all_models:
                result_current = [
                    result
                    for result in results
                    if (result[DICT_DATASET] == dataset) and
                       (result[DICT_MODEL] == model) and
                       (result[DICT_FEATURE_LABEL] == enrichment)
                ][0]
                result_current_recall = result_current[DICT_RECALL]
                if result_current_recall > max_current:
                    max_current = result_current_recall

            for model in all_models:
                result_current = [
                    result
                    for result in results
                    if (result[DICT_DATASET] == dataset) and
                       (result[DICT_MODEL] == model) and
                       (result[DICT_FEATURE_LABEL] == enrichment)
                ][0]
                result_current_recall = result_current[DICT_RECALL]
                if result_current_recall == max_current:
                    counts[enrichment] += 1

    max_value = max(counts.values())
    max_items = {k: v for k, v in counts.items() if v == max_value}

    print(f"    - By Recall: {max_items}")
    return set(max_items.keys())


def find_enrichment_through_auc_roc_for_model(model):
    results = get_results_of_some_models([model], all_results)

    counts = Counter(all_enrichments)
    counts = dict(counts)
    for k, v in counts.items():
        counts[k] = 0

    for dataset in all_datasets:
        max_current = 0

        for enrichment in all_enrichments:
            result_current = [
                result
                for result in results
                if (result[DICT_DATASET] == dataset) and
                   (result[DICT_MODEL] == model) and
                   (result[DICT_FEATURE_LABEL] == enrichment)
            ][0]
            result_current_auc_roc = abs(0.5 - result_current[DICT_AUC_ROC])
            if result_current_auc_roc > max_current:
                max_current = result_current_auc_roc

        for enrichment in all_enrichments:
            result_current = [
                result
                for result in results
                if (result[DICT_DATASET] == dataset) and
                   (result[DICT_MODEL] == model) and
                   (result[DICT_FEATURE_LABEL] == enrichment)
            ][0]
            result_current_auc_roc = abs(0.5 - result_current[DICT_AUC_ROC])
            if result_current_auc_roc == max_current:
                counts[enrichment] += 1

    max_value = max(counts.values())
    max_items = {k: v for k, v in counts.items() if v == max_value}

    print(f"    - By AUC-ROC: {max_items}")
    return set(max_items.keys())


def find_enrichment_through_precision_for_model(model):
    results = get_results_of_some_models([model], all_results)

    counts = Counter(all_enrichments)
    counts = dict(counts)
    for k, v in counts.items():
        counts[k] = 0

    for dataset in all_datasets:
        max_current = -1

        for enrichment in all_enrichments:
            result_current = [
                result
                for result in results
                if (result[DICT_DATASET] == dataset) and
                   (result[DICT_MODEL] == model) and
                   (result[DICT_FEATURE_LABEL] == enrichment)
            ][0]
            result_current_auc_roc = result_current[DICT_PRECISION]
            if result_current_auc_roc > max_current:
                max_current = result_current_auc_roc

        for enrichment in all_enrichments:
            result_current = [
                result
                for result in results
                if (result[DICT_DATASET] == dataset) and
                   (result[DICT_MODEL] == model) and
                   (result[DICT_FEATURE_LABEL] == enrichment)
            ][0]
            result_current_auc_roc = result_current[DICT_PRECISION]
            if result_current_auc_roc == max_current:
                counts[enrichment] += 1

    max_value = max(counts.values())
    max_items = {k: v for k, v in counts.items() if v == max_value}

    print(f"    - By Precision: {max_items}")
    return set(max_items.keys())


def find_enrichment_through_recall_for_model(model):
    results = get_results_of_some_models([model], all_results)

    counts = Counter(all_enrichments)
    counts = dict(counts)
    for k, v in counts.items():
        counts[k] = 0

    for dataset in all_datasets:
        max_current = -1

        for enrichment in all_enrichments:
            result_current = [
                result
                for result in results
                if (result[DICT_DATASET] == dataset) and
                   (result[DICT_MODEL] == model) and
                   (result[DICT_FEATURE_LABEL] == enrichment)
            ][0]
            result_current_auc_roc = result_current[DICT_RECALL]
            if result_current_auc_roc > max_current:
                max_current = result_current_auc_roc

        for enrichment in all_enrichments:
            result_current = [
                result
                for result in results
                if (result[DICT_DATASET] == dataset) and
                   (result[DICT_MODEL] == model) and
                   (result[DICT_FEATURE_LABEL] == enrichment)
            ][0]
            result_current_auc_roc = result_current[DICT_RECALL]
            if result_current_auc_roc == max_current:
                counts[enrichment] += 1

    max_value = max(counts.values())
    max_items = {k: v for k, v in counts.items() if v == max_value}

    print(f"    - By Recall: {max_items}")
    return set(max_items.keys())


def is_enrichment_1_better_then_enrichment_2_through_auc_roc(enrichment_1, enrichment_2):
    results = get_results_of_some_enrichments([enrichment_1, enrichment_2], all_results)

    counts = Counter([enrichment_1, enrichment_2])
    counts = dict(counts)
    for k, v in counts.items():
        counts[k] = 0

    for dataset in all_datasets:
        for model in all_models:
            result_current_1 = [
                result
                for result in results
                if (result[DICT_DATASET] == dataset) and
                   (result[DICT_MODEL] == model) and
                   (result[DICT_FEATURE_LABEL] == enrichment_1)
            ][0]
            result_current_auc_roc_1 = abs(0.5 - result_current_1[DICT_RECALL])

            result_current_2 = [
                result
                for result in results
                if (result[DICT_DATASET] == dataset) and
                   (result[DICT_MODEL] == model) and
                   (result[DICT_FEATURE_LABEL] == enrichment_2)
            ][0]
            result_current_auc_roc_2 = abs(0.5 - result_current_2[DICT_RECALL])

            if result_current_auc_roc_1 > result_current_auc_roc_2:
                counts[enrichment_1] += 1
            else:
                if result_current_auc_roc_1 == result_current_auc_roc_2:
                    counts[enrichment_1] += 1
                    counts[enrichment_2] += 1
                else:
                    counts[enrichment_2] += 1

    max_value = max(counts.values())
    max_items = {k: v for k, v in counts.items() if v == max_value}

    print(f"    - By AUC-ROC: {max_items}")
    return set(max_items.keys())


def is_enrichment_1_better_then_enrichment_2_through_precision(enrichment_1, enrichment_2):
    results = get_results_of_some_enrichments([enrichment_1, enrichment_2], all_results)

    counts = Counter([enrichment_1, enrichment_2])
    counts = dict(counts)
    for k, v in counts.items():
        counts[k] = 0

    for dataset in all_datasets:
        for model in all_models:
            result_current_1 = [
                result
                for result in results
                if (result[DICT_DATASET] == dataset) and
                   (result[DICT_MODEL] == model) and
                   (result[DICT_FEATURE_LABEL] == enrichment_1)
            ][0]
            result_current_auc_roc_1 = result_current_1[DICT_PRECISION]

            result_current_2 = [
                result
                for result in results
                if (result[DICT_DATASET] == dataset) and
                   (result[DICT_MODEL] == model) and
                   (result[DICT_FEATURE_LABEL] == enrichment_2)
            ][0]
            result_current_auc_roc_2 = result_current_2[DICT_PRECISION]

            if result_current_auc_roc_1 > result_current_auc_roc_2:
                counts[enrichment_1] += 1
            else:
                if result_current_auc_roc_1 == result_current_auc_roc_2:
                    counts[enrichment_1] += 1
                    counts[enrichment_2] += 1
                else:
                    counts[enrichment_2] += 1

    max_value = max(counts.values())
    max_items = {k: v for k, v in counts.items() if v == max_value}

    print(f"    - By Precision: {max_items}")
    return set(max_items.keys())


def is_enrichment_1_better_then_enrichment_2_through_recall(enrichment_1, enrichment_2):
    results = get_results_of_some_enrichments([enrichment_1, enrichment_2], all_results)

    counts = Counter([enrichment_1, enrichment_2])
    counts = dict(counts)
    for k, v in counts.items():
        counts[k] = 0

    for dataset in all_datasets:
        for model in all_models:
            result_current_1 = [
                result
                for result in results
                if (result[DICT_DATASET] == dataset) and
                   (result[DICT_MODEL] == model) and
                   (result[DICT_FEATURE_LABEL] == enrichment_1)
            ][0]
            result_current_auc_roc_1 = result_current_1[DICT_RECALL]

            result_current_2 = [
                result
                for result in results
                if (result[DICT_DATASET] == dataset) and
                   (result[DICT_MODEL] == model) and
                   (result[DICT_FEATURE_LABEL] == enrichment_2)
            ][0]
            result_current_auc_roc_2 = result_current_2[DICT_RECALL]

            if result_current_auc_roc_1 > result_current_auc_roc_2:
                counts[enrichment_1] += 1
            else:
                if result_current_auc_roc_1 == result_current_auc_roc_2:
                    counts[enrichment_1] += 1
                    counts[enrichment_2] += 1
                else:
                    counts[enrichment_2] += 1

    max_value = max(counts.values())
    max_items = {k: v for k, v in counts.items() if v == max_value}

    print(f"    - By Recall: {max_items}")
    return set(max_items.keys())


def is_learned_enrichment_better_than_precomputed_enrichment_through_auc_roc(datasets=all_datasets):
    results = get_results_of_some_datasets(datasets, all_results)

    counts = Counter(["learned", "precomputed"])
    counts = dict(counts)
    for k, v in counts.items():
        counts[k] = 0

    for dataset in datasets:
        for model in all_models:
            result_current = [
                result
                for result in results
                if (result[DICT_DATASET] == dataset) and
                   (result[DICT_MODEL] == model)]
            result_current_auc_roc = [
                abs(0.5 - result[DICT_AUC_ROC])
                for result in result_current]

            maximum = max(result_current_auc_roc)
            for result in result_current:
                if abs(0.5 - result[DICT_AUC_ROC]) == maximum:
                    if result[DICT_FEATURE_LABEL] in LEARNED_ENRICHMENTS:
                        counts["learned"] += 1
                    if result[DICT_FEATURE_LABEL] in PRECOMPUTED_ENRICHMENTS:
                        counts["precomputed"] += 1

    max_value = max(counts.values())
    max_items = {k: v for k, v in counts.items() if v == max_value}

    print(f"    - By AUC-ROC: {max_items}")
    return set(max_items.keys())


def is_learned_enrichment_better_than_precomputed_enrichment_through_precision(datasets=all_datasets):
    results = get_results_of_some_datasets(datasets, all_results)

    counts = Counter(["learned", "precomputed"])
    counts = dict(counts)
    for k, v in counts.items():
        counts[k] = 0

    for dataset in datasets:
        for model in all_models:
            result_current = [
                result
                for result in results
                if (result[DICT_DATASET] == dataset) and
                   (result[DICT_MODEL] == model)]
            result_current_auc_roc = [
                result[DICT_PRECISION]
                for result in result_current]

            maximum = max(result_current_auc_roc)
            for result in result_current:
                if result[DICT_PRECISION] == maximum:
                    if result[DICT_FEATURE_LABEL] in LEARNED_ENRICHMENTS:
                        counts["learned"] += 1
                    if result[DICT_FEATURE_LABEL] in PRECOMPUTED_ENRICHMENTS:
                        counts["precomputed"] += 1

    max_value = max(counts.values())
    max_items = {k: v for k, v in counts.items() if v == max_value}

    print(f"    - By Precision: {max_items}")
    return set(max_items.keys())


def is_learned_enrichment_better_than_precomputed_enrichment_through_recall(datasets=all_datasets):
    results = get_results_of_some_datasets(datasets, all_results)

    counts = Counter(["learned", "precomputed"])
    counts = dict(counts)
    for k, v in counts.items():
        counts[k] = 0

    for dataset in datasets:
        for model in all_models:
            result_current = [
                result
                for result in results
                if (result[DICT_DATASET] == dataset) and
                   (result[DICT_MODEL] == model)]
            result_current_auc_roc = [
                result[DICT_RECALL]
                for result in result_current]

            maximum = max(result_current_auc_roc)
            for result in result_current:
                if result[DICT_RECALL] == maximum:
                    if result[DICT_FEATURE_LABEL] in LEARNED_ENRICHMENTS:
                        counts["learned"] += 1
                    if result[DICT_FEATURE_LABEL] in PRECOMPUTED_ENRICHMENTS:
                        counts["precomputed"] += 1

    max_value = max(counts.values())
    max_items = {k: v for k, v in counts.items() if v == max_value}

    print(f"    - By Recall: {max_items}")
    return set(max_items.keys())


def find_best_enrichment_for_domain(datasets, domain_name):
    print("Best enrichment for " + domain_name + ":")
    set_1 = find_enrichment_through_auc_roc_for_domain(datasets)
    has_error_1 = any(item.startswith("Attr + Error") for item in set_1)
    has_emd_1 = any(item.startswith("Attr + Emd") for item in set_1)
    has_str_1 = any(item.startswith("Attr + Str") for item in set_1)
    set_2 = find_enrichment_through_precision_for_domain(datasets)
    has_error_2 = any(item.startswith("Attr + Error") for item in set_2)
    has_emd_2 = any(item.startswith("Attr + Emd") for item in set_2)
    has_str_2 = any(item.startswith("Attr + Str") for item in set_2)
    set_3 = find_enrichment_through_recall_for_domain(datasets)
    has_error_3 = any(item.startswith("Attr + Error") for item in set_3)
    has_emd_3 = any(item.startswith("Attr + Emd") for item in set_3)
    has_str_3 = any(item.startswith("Attr + Str") for item in set_3)

    set_all = set_1 & set_2 & set_3

    print("Result: " + str(set_all))
    if len(set_all) == 0:
        if has_error_1 and has_error_2 and has_error_3:
            print("Additional result: Error")
        if has_str_1 and has_str_2 and has_str_3:
            print("Additional result: Str")
        if has_emd_1 and has_emd_2 and has_emd_3:
            print("Additional result: Emd")
    print("\n\n")


def find_best_enrichment_for_model(model):
    print("Best enrichment for " + model + ":")
    set_1 = find_enrichment_through_auc_roc_for_model(model)
    has_error_1 = any(item.startswith("Attr + Error") for item in set_1)
    has_emd_1 = any(item.startswith("Attr + Emd") for item in set_1)
    has_str_1 = any(item.startswith("Attr + Str") for item in set_1)
    set_2 = find_enrichment_through_precision_for_model(model)
    has_error_2 = any(item.startswith("Attr + Error") for item in set_2)
    has_emd_2 = any(item.startswith("Attr + Emd") for item in set_2)
    has_str_2 = any(item.startswith("Attr + Str") for item in set_2)
    set_3 = find_enrichment_through_recall_for_model(model)
    has_error_3 = any(item.startswith("Attr + Error") for item in set_3)
    has_emd_3 = any(item.startswith("Attr + Emd") for item in set_3)
    has_str_3 = any(item.startswith("Attr + Str") for item in set_3)

    set_all = set_1 & set_2 & set_3

    print("Result: " + str(set_all))
    if len(set_all) == 0:
        if has_error_1 and has_error_2 and has_error_3:
            print("Additional result: Error")
        if has_str_1 and has_str_2 and has_str_3:
            print("Additional result: Str")
        if has_emd_1 and has_emd_2 and has_emd_3:
            print("Additional result: Emd")
    print("\n\n")


def is_enrichment_1_better_then_enrichment_2(enrichment_1, enrichment_2):
    print(f"If enrichment {enrichment_1} is better than enrichment {enrichment_2}?")
    set_1 = is_enrichment_1_better_then_enrichment_2_through_auc_roc(enrichment_1, enrichment_2)
    set_2 = is_enrichment_1_better_then_enrichment_2_through_recall(enrichment_1, enrichment_2)
    set_3 = is_enrichment_1_better_then_enrichment_2_through_precision(enrichment_1, enrichment_2)

    set_all = set_1 & set_2 & set_3

    print("Result: " + str(set_all))
    if (len(set_all) == 2) or (len(set_all) == 0):
        print("Not yes, nor no")
        print("\n\n")
        return 0
    else:
        if (len(set_all) == 1) and (len(set_all & set([enrichment_1])) == 1):
            print("Yes")
            print("\n\n")
            return 1
        else:
            print("No")
            print("\n\n")
            return -1


def is_learned_enrichment_better_than_precomputed_enrichment_for_group(datasets, group_name):
    print(f"If learned enrichment is better than precomputed enrichment in {group_name}?")
    set_1 = is_learned_enrichment_better_than_precomputed_enrichment_through_auc_roc(datasets)
    set_2 = is_learned_enrichment_better_than_precomputed_enrichment_through_recall(datasets)
    set_3 = is_learned_enrichment_better_than_precomputed_enrichment_through_precision(datasets)

    set_all = set_1 & set_2 & set_3

    print("Result: " + str(set_all))
    if len(set_all) == 0:
        print("Not yes, nor no")
    else:
        has_learned = any(item.startswith("learned") for item in set_all)
        has_precomputed = any(item.startswith("precomputed") for item in set_all)
        if has_learned and has_precomputed:
            print("Both")
        else:
            if has_learned:
                print("Yes")
            else:
                print("No")
    print("\n\n")


def is_learned_enrichment_better_than_precomputed_enrichment():
    set_1 = is_learned_enrichment_better_than_precomputed_enrichment_through_auc_roc()
    set_2 = is_learned_enrichment_better_than_precomputed_enrichment_through_recall()
    set_3 = is_learned_enrichment_better_than_precomputed_enrichment_through_precision()

    set_all = set_1 & set_2 & set_3

    print("Result: " + str(set_all))
    if len(set_all) == 0:
        print("Not yes, nor no")
    else:
        has_learned = any(item.startswith("learned") for item in set_all)
        has_precomputed = any(item.startswith("precomputed") for item in set_all)
        if has_learned and has_precomputed:
            print("Both")
        else:
            if has_learned:
                print("Yes")
            else:
                print("No")
    print("\n\n")


def main():
    print("Checking")

    # find_best_enrichment_for_domain(DATASETS_SOCIAL_NETWORKS, 'social_networks') # Result: {'Attr + Error2'}
    # find_best_enrichment_for_domain(DATASETS_CO_PURCHASE, 'co_purchase') # Not given
    # find_best_enrichment_for_domain(DATASETS_CITATION_NETWORKS, 'citation_networks') # Additional result: Str
    # find_best_enrichment_for_domain(DATASETS_USER_SUBREDDIT, 'user_subreddit') # Not given
    # find_best_enrichment_for_domain(DATASETS_UNDER_SAME_HASHTAG, 'under_same_hashtag') # Result: {'Attr + Str', 'Attr + Emd2'}
    # find_best_enrichment_for_domain(DATASETS_WORK_COLLABORATION, 'work_collaboration') # Result: {'Attr + Emd2'}

    # find_best_enrichment_for_domain(ONLY_ATTR_ANOMALY_DATASETS,
    # 'ONLY_ATTR_ANOMALY_DATASETS')  # Result: {'Attr + Emd2'}
    # find_best_enrichment_for_domain(STR_OR_ATTR_ANOMALY_DATASET,
    # 'STR_OR_ATTR_ANOMALY_DATASET')  # Result: {'Attr + Error2'}
    # find_best_enrichment_for_domain(STR_AND_ATTR_ANOMALY_DATASETS,
    # 'STR_AND_ATTR_ANOMALY_DATASETS')  # Result: {'Attr + Str'}

    # find_best_enrichment_for_model("AnomalyDAE") # Result: {'Attr + Error2'}
    # find_best_enrichment_for_model("CoLA") # Result: {'Attr + Error2'}
    # find_best_enrichment_for_model("OCGNN") # Result: {'Attr + Str'}

    # is_enrichment_1_better_then_enrichment_2(FEATURE_LABEL_STR, FEATURE_LABEL_STR2) # Not given
    # is_enrichment_1_better_then_enrichment_2(FEATURE_LABEL_STR, FEATURE_LABEL_STR3) # Yes
    # is_enrichment_1_better_then_enrichment_2(FEATURE_LABEL_STR2, FEATURE_LABEL_STR3) # Not given

    # is_enrichment_1_better_then_enrichment_2(FEATURE_LABEL_ERROR1, FEATURE_LABEL_ERROR2) # Not given

    # is_enrichment_1_better_then_enrichment_2(FEATURE_LABEL_EMD1, FEATURE_LABEL_EMD2)  # Not given

    # is_enrichment_1_better_then_enrichment_2(FEATURE_LABEL_STR, FEATURE_LABEL_STANDARD) # Not given
    # is_enrichment_1_better_then_enrichment_2(FEATURE_LABEL_STR2, FEATURE_LABEL_STANDARD) # Not given
    # is_enrichment_1_better_then_enrichment_2(FEATURE_LABEL_STR3, FEATURE_LABEL_STANDARD) # Not given
    # is_enrichment_1_better_then_enrichment_2(FEATURE_LABEL_EMD1, FEATURE_LABEL_STANDARD) # Yes
    # is_enrichment_1_better_then_enrichment_2(FEATURE_LABEL_EMD2, FEATURE_LABEL_STANDARD) # Yes
    # is_enrichment_1_better_then_enrichment_2(FEATURE_LABEL_ERROR1, FEATURE_LABEL_STANDARD) # Not given
    # is_enrichment_1_better_then_enrichment_2(FEATURE_LABEL_ERROR2, FEATURE_LABEL_STANDARD) # Not given

    # is_learned_enrichment_better_than_precomputed_enrichment()  # Yes

    # is_learned_enrichment_better_than_precomputed_enrichment_for_group(DATASETS_SOCIAL_NETWORKS,
    #                                                                    'social_networks')  # Yes
    # is_learned_enrichment_better_than_precomputed_enrichment_for_group(DATASETS_CO_PURCHASE,
    #                                                                    'co_purchase')  # Yes
    # is_learned_enrichment_better_than_precomputed_enrichment_for_group(DATASETS_CITATION_NETWORKS,
    #                                                                    'citation_networks')  # No
    # is_learned_enrichment_better_than_precomputed_enrichment_for_group(DATASETS_USER_SUBREDDIT,
    #                                                                    'user_subreddit')  # Not given
    # is_learned_enrichment_better_than_precomputed_enrichment_for_group(DATASETS_UNDER_SAME_HASHTAG,
    #                                                                    'under_same_hashtag')  # Yes
    # is_learned_enrichment_better_than_precomputed_enrichment_for_group(DATASETS_WORK_COLLABORATION,
    #                                                                    'work_collaboration')  # Yes

    # is_learned_enrichment_better_than_precomputed_enrichment_for_group(ONLY_ATTR_ANOMALY_DATASETS,
    #                                                                    'ONLY_ATTR_ANOMALY_DATASETS')  # Yes
    # is_learned_enrichment_better_than_precomputed_enrichment_for_group(STR_OR_ATTR_ANOMALY_DATASET,
    #                                                                    'STR_OR_ATTR_ANOMALY_DATASET')  # Yes
    # is_learned_enrichment_better_than_precomputed_enrichment_for_group(STR_AND_ATTR_ANOMALY_DATASETS,
    #                                                                    'STR_AND_ATTR_ANOMALY_DATASETS')  # Yes


if __name__ == "__main__":
    main()
