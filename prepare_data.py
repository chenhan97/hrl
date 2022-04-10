import json
from sentence_transformers import SentenceTransformer
from hrlmodel.hrl_model import SentenceEncodeModule, HigherReinforceModule, LowerReinforceModule
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("dataset/ter_mul/4/train.json") as infile, open("dataset/ter_mul/4/train2.json", mode='w') as outfile:
    data = json.load(infile)
    json_list = []
    for d_no, d in enumerate(data):
        print(d_no)
        extend_instance_dict = d
        if '.' in d['stanford_pos']:
            sent_end_loc_list = [idx for (idx, pos) in enumerate(d['stanford_pos']) if pos == '.']
        else:
            sent_end_loc_list = [len(d['stanford_pos'])-1]
        split_sents = []
        for i in range(len(sent_end_loc_list)):
            if i == 0:
                sent = " ".join(d['token'][0:sent_end_loc_list[i]])
            else:
                sent = " ".join(d['token'][sent_end_loc_list[i-1]:sent_end_loc_list[i]])
            split_sents.append(sent)
        sent_embs = model.encode(split_sents)
        sent_embs = sent_embs.tolist()
        extend_instance_dict["sent_emb"] = sent_embs
        extend_instance_dict["split_idx"] = sent_end_loc_list
        json_list.append(extend_instance_dict)
    json.dump(json_list, outfile)
