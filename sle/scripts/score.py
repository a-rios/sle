import argparse
import re
import csv
import pandas as pd
import numpy as np
from sle.scorer import SLEScorer
from sklearn.metrics import mean_absolute_error

def print_single(sentences, results):
    for sentence, score in zip(sentences, results['sle']):
            print(f"{score}\t{sentence}")

def print_parallel(sentences, src_sentences, results):
    for sentence, src_sentence, score, delta in zip(sentences, src_sentences, results['sle'], results['sle_delta']):
            print(f"{score}\t{delta}\t{src_sentence}\t{sentence}")

def print_doc_mae(preds, labels, doc_ids):
    doc_preds = {}
    doc_labs = {}
    for i in range(len(doc_ids)):
        if doc_ids[i] not in doc_preds:
            doc_preds[doc_ids[i]] = [preds[i]]
            doc_labs[doc_ids[i]] = [labels[i]]
        else:
            doc_preds[doc_ids[i]].append(preds[i])
            doc_labs[doc_ids[i]].append(labels[i])
    doc_means = []
    doc_gts = []
    for k, v in doc_preds.items():
        doc_means.append(np.mean(v))
        doc_gts.append(np.mean(doc_labs[k]))
    print(f"doc mae: {mean_absolute_error(doc_gts, doc_means)}")

def score_text(scorer, batch_size, text, src_text=None):
    with open(text, 'r') as f:
        sentences = [sentence.rstrip() for sentence in f.readlines()]
    if src_text is not None:
        with open(src_text, 'r') as f:
            src_sentences = [sentence.rstrip() for sentence in f.readlines()]
    else:
        src_sentences=None
    results = scorer.score(sentences, inputs=src_sentences, batch_size=batch_size)

    if src_sentences is not None:
        print_parallel(sentences, src_sentences, results)
    else:
        print_single(sentences, results)


def score_csv(scorer, batch_size, in_csv, id_column_name, label_colum_name, smooth_label_colum_name, text_colum_name, source_text_colum_name=None, doc_mae=False):
    df = pd.read_csv(in_csv)
    sentences = df[text_colum_name]
    smoothed = df[smooth_label_colum_name]
    labels = df[label_colum_name]
    doc_ids = df[id_column_name]
    if source_text_colum_name is not None:
        src_sentences = df[source_text_colum_name]
    else:
        src_sentences = None

    results = scorer.score(sentences, inputs=src_sentences)

    if src_sentences is not None:
        print_parallel(sentences, src_sentences, results)
    else:
        print_single(sentences, results)

    # doc_mae
    if args.doc_mae:
        print_doc_mae(results['sle'], labels, doc_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--text", type=str, help="text file to score (one sentence per line)", metavar="PATH")
    parser.add_argument("--src_text", type=str, help="source text file to calculate delta (optional)", metavar="PATH")
    parser.add_argument("--csv", type=str, help="csv to score (with doc ids + labels)", metavar="PATH")
    parser.add_argument("--model", type=str, help="path to folder containing scorer model", metavar="PATH", required=True)
    parser.add_argument("--id_column_name", type=str, default="doc_id", help="column name for doc ids in csv")
    parser.add_argument("--label_colum_name", type=str, default="y", help="column name for labels in csv")
    parser.add_argument("--smooth_label_colum_name", type=str, default="y_smooth", help="column name for smoothed labels in csv")
    parser.add_argument("--text_colum_name", type=str, default="text", help="column name for text in csv")
    parser.add_argument("--source_text_colum_name", type=str, default=None, help="column name for source text in csv (if present)")
    parser.add_argument("--doc_mae", action="store_true", help="print document level mean absolute error (needs csv input with labels)")

    args = parser.parse_args()

    scorer = SLEScorer(args.model)
    if args.text:
        score_text(scorer, args.batch_size, args.text, args.src_text)
    elif args.csv:
        score_csv(scorer, args.batch_size, args.csv, args.id_column_name, args.label_colum_name, args.smooth_label_colum_name, args.text_colum_name, args.source_text_colum_name, args.doc_mae)
