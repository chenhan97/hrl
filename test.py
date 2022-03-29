import json
import spacy
from sentence_transformers import SentenceTransformer
from hrlmodel.hrl_model import SentenceEncodeModule, HigherReinforceModule, LowerReinforceModule
import torch

nlp = spacy.load("en_core_web_trf")
model = SentenceTransformer('all-MiniLM-L6-v2')

with open("dataset/test.json") as infile, open("dataset/test2.json", mode='w') as outfile:
    data = json.load(infile)
    json_list = []
    for d_no, d in enumerate(data):
        extend_instance_dict = d
        sents = " ".join(d['token'])
        doc = nlp(sents)
        split_sents = []
        for sent in doc.sents:
            split_sents.append(sent.text)
        sent_embs = model.encode(split_sents)
        extend_instance_dict["sent_emb"] = sent_embs

    #json.dump(json_string, outfile)


