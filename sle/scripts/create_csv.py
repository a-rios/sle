import argparse
import re
from mosestokenizer import MosesDetokenizer
import csv
import json
from bs4 import BeautifulSoup
from somajo import SoMaJo
tokenizer = SoMaJo("de_CMC")
doc_xml_key = "document_xml|||PandasExtractor|||https://github.com/ZurichNLP/ats-extraction-pipeline|||91b332d88497749054cd821bcc3583e7e1784a53"
# %%
def remove_xml(line):
    return BeautifulSoup(line, "html.parser").text

# %%
# def clean_paragraphs(paragraphs):
#     new = []
#     for p in paragraphs:
#         p = remove_xml(p)
#         p = re.sub(r"\\n", " ", p)
#         p = re.sub(" +", " ", p)
#         p = p.strip()
#         new.append(p)
#     return [p for p in new if p]

def clean_xml(p):
    p = remove_xml(p)
    p = re.sub(r"\\n", " ", p)
    p = re.sub(" +", " ", p)
    p = p.strip()
    return p

from somajo import SoMaJo
tokenizer = SoMaJo("de_CMC")

def detokenize(tokens):
    sent = ""
    for token in tokens:
        if token.space_after:
            sent += token.text + " "
        else:
            sent += token.text
    return sent

def split_sentences(paragraph):
    sentences = tokenizer.tokenize_text([paragraph])
    return [detokenize(s) for s in sentences]


def write_csv(samples, out_csv):
    with open(out_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id","text","y"])
        for key, sample in samples.items():
            writer.writerow(sample)

def capito_csv(infile, out_csv):

    map_labels = {"original": 0,
                  "b1" : 1,
                  "a2" : 2,
                  "a1" : 3}

    with open(infile) as f:
        data = json.load(f)

    samples = dict()
    for doc in data["documents"]:
        assert (doc["metadata"]["identifier"] == doc["metadata"]["foreign_system_id"])
        doc_id = doc["metadata"]["identifier"]

        # get sentences
        for level in doc["segmentations"].keys():
            assert (len(doc["segmentations"][level][doc_xml_key])) == 1
            documents = [x["content"] for x in doc["segmentations"][level][doc_xml_key]]

            for document in documents:
                text = clean_xml(document)
                sentences = split_sentences(text)
                label = map_labels[level]
                for sentence in sentences:
                    sample_key = f"{doc_id},{sentence},{label}"
                    samples[sample_key] = [doc_id, sentence, label] # dict to avoid duplicates
        write_csv(samples, out_csv)




def newsela_csv(bitext, out_csv):

    with open(bitext, 'r') as f, MosesDetokenizer('en') as detokenize:
        # DOC1	V0	V3	Poll finds Americans OK with women in combat	Women get the OK to fight in combat units
        samples = dict() # use dict to avoid duplicates
        for line in f.readlines():
            doc_id, complex_level, simple_level, complex_sent, simple_sent = line.rstrip().split('\t')
            complex_level = re.sub('^V', '', complex_level)
            simple_level = re.sub('^V', '', simple_level)
            sample1 = f"{doc_id},{detokenize(complex_sent.split(' '))},{complex_level}"
            sample2 = f"{doc_id},{detokenize(simple_sent.split(' '))},{simple_level}"
            samples[sample1] =  [doc_id,detokenize(complex_sent.split(' ')),complex_level]
            samples[sample2] =  [doc_id,detokenize(simple_sent.split(' ')),simple_level]

    write_csv(samples, out_csv)
   # with open(out_csv, 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["doc_id","text","y"])
    #     for key, sample in samples.items():
    #         writer.writerow(sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, help="newsela bitext", metavar="PATH", required=True)
    parser.add_argument("--out-csv", type=str, help="newsela bitext", metavar="PATH", required=True)
    parser.add_argument("--corpus", type=str, help="newsela or capito", required=True)


    args = parser.parse_args()
    if args.corpus == "newsela":
        newsela_csv(args.infile, args.out_csv)
    elif args.corpus == "capito":
        capito_csv(args.infile, args.out_csv)
    else:
        print("Undefined corpus, set to either 'newsela' or 'capito'")
