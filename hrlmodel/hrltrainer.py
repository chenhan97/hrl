import torch
import torch.nn as nn
from hrlmodel.hrl_model import SentenceEncodeModule, HigherReinforceModule, LowerReinforceModule
import numpy as np


def adjust_stanford_head(old_head, select_sent):
    sum_prior_selected_tokens = 0
    mapping_dict = {}
    for i in range(len(select_sent)):
        if i!=0 and select_sent[i][0] == select_sent[i-1][1]:
            for j in range(select_sent[i][0]+1, select_sent[i][1]+1):
                mapping_dict[j+1] = j - select_sent[i][0] + sum_prior_selected_tokens
            sum_prior_selected_tokens += select_sent[i][1] - select_sent[i][0]
        else:
            for j in range(select_sent[i][0], select_sent[i][1]+1):
                mapping_dict[j+1] = j - select_sent[i][0] + sum_prior_selected_tokens + 1
            sum_prior_selected_tokens += select_sent[i][1] - select_sent[i][0] + 1
    old_head_list = old_head.tolist()
    new_head = [mapping_dict[i] if i in mapping_dict.keys() else i for i in old_head_list]
    new_head = [-1 if i>len(new_head) else i for i in new_head]
    if 0 not in new_head:
        new_head[0] = 0
    new_head = torch.LongTensor(new_head)
    return new_head


def monte_carlo_sample(re_model_trainer, sent_idx_list, sent_idx, batch, batch_idx, mode, sample_rate=3):
    split_idx = batch[12][batch_idx]
    sent_idx_list_with_sample = sent_idx_list.copy()
    # sampling
    if len(split_idx) > (sent_idx + 1):
        sampled_sents_seed = np.random.randint(0, 2, len(split_idx) - sent_idx - 1)
        for i in range(len(sampled_sents_seed)):
            if sampled_sents_seed[i] == 1:
                sent_idx_list_with_sample.append(sent_idx + i + 1)
    select_sent = [[split_idx[i - 1], split_idx[i]] if i != 0 else [0, split_idx[i]] for i in sent_idx_list_with_sample]
    new_words = torch.LongTensor([0])
    new_pos = torch.LongTensor([0])
    new_head = torch.LongTensor([0])
    new_deprel = torch.LongTensor([0])
    new_first_positions = torch.LongTensor([0])
    new_second_positions = torch.LongTensor([0])
    new_third_positions = torch.LongTensor([0])
    for i in range(len(select_sent)):
        new_words = torch.cat((new_words, batch[0][batch_idx][select_sent[i][0]:select_sent[i][1] + 1]))
        new_pos = torch.cat((new_pos, batch[2][batch_idx][select_sent[i][0]:select_sent[i][1] + 1]))
        new_deprel = torch.cat((new_deprel, batch[3][batch_idx][select_sent[i][0]:select_sent[i][1] + 1]))
        new_head = torch.cat((new_head, batch[4][batch_idx][select_sent[i][0]:select_sent[i][1] + 1]))
        new_first_positions = torch.cat(
            (new_first_positions, batch[5][batch_idx][select_sent[i][0]:select_sent[i][1] + 1]))
        new_second_positions = torch.cat(
            (new_second_positions, batch[6][batch_idx][select_sent[i][0]:select_sent[i][1] + 1]))
        new_third_positions = torch.cat(
            (new_third_positions, batch[7][batch_idx][select_sent[i][0]:select_sent[i][1] + 1]))
    new_head = adjust_stanford_head(new_head[1:], select_sent).unsqueeze(dim=0)
    new_words = new_words[1:].unsqueeze(dim=0)
    new_pos = new_pos[1:].unsqueeze(dim=0)
    new_deprel = new_deprel[1:].unsqueeze(dim=0)
    new_first_positions = new_first_positions[1:].unsqueeze(dim=0)
    new_second_positions = new_second_positions[1:].unsqueeze(dim=0)
    new_third_positions = new_third_positions[1:].unsqueeze(dim=0)
    new_masks = torch.eq(new_words, 0)
    if mode:
        preds, _, loss = re_model_trainer.predict((new_words, new_masks, new_pos, new_deprel, new_head, new_first_positions,
                                               new_second_positions, new_third_positions, batch[8][batch_idx],
                                               batch[9][batch_idx].unsqueeze(dim=0), batch[10][batch_idx]),
                                              unsort=False)
        return loss
    else:
        return (new_words, new_masks, new_pos, new_deprel, new_head, new_first_positions,
                                               new_second_positions, new_third_positions, batch[8][batch_idx],
                                               batch[9][batch_idx].unsqueeze(dim=0), batch[10][batch_idx])


class HRLTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.sentence_encoder = SentenceEncodeModule()
        self.HR = HigherReinforceModule(64, 64)
        self.LR = LowerReinforceModule(64)
        if opt['cuda']:
            self.HR = self.HR.cuda()
            self.LR = self.LR.cuda()
            self.sentence_encoder = self.sentence_encoder.cuda()
        self.optimizer = torch.optim.Adam(
            list(self.HR.parameters()) + list(self.LR.parameters()) + list(self.sentence_encoder.parameters()),
            lr=0.001)

    def update(self, batch, re_model):

        # step forward
        sent_emb, split_idx, rels = batch[11], batch[12], batch[9]
        self.HR.train()
        self.LR.train()
        self.sentence_encoder.train()
        self.optimizer.zero_grad()
        expect = 0.0
        for i in range(len(sent_emb)):
            instance_sent_emb = sent_emb[i]
            instance_sent_emb = torch.Tensor(instance_sent_emb)
            if self.opt['cuda']:
                instance_sent_emb = instance_sent_emb.cuda()
            sent_encoding = self.sentence_encoder(instance_sent_emb)
            for sent in sent_encoding:
                prob = self.HR(torch.unsqueeze(sent, dim=0), rels[i])
                option = np.random.choice([0, 1], size=1, p=[1 - prob.item(), prob.item()])[0]
                if option == 1:
                    # keep the index of LR-selected sentences

                    # TESTING!! disable LR
                    '''sent_idx_list, sent_idx = [], 0
                    for sent_ in sent_encoding:
                        prob_ = self.LR(torch.unsqueeze(sent_, dim=0))
                        action = np.random.choice([0, 1], size=1, p=[1 - prob_.item(), prob_.item()])[0]
                        if action == 1:
                            sent_idx_list.append(sent_idx)
                        sent_idx += 1
                        if len(sent_idx_list) == 0:
                            reward = 10000000000
                        else:
                            reward = monte_carlo_sample(re_model, sent_idx_list, sent_idx, batch, i, True)
                    expect += prob * reward + prob_ * reward'''
                    sent_idx_list = [i for i in range(len(sent_encoding))]
                    batch0 = batch[0][i].numpy()
                    batch0 = np.trim_zeros(batch0,trim='b')
                    batch0 = torch.LongTensor(batch0).unsqueeze(dim=0)
                    batch2 = batch[2][i].numpy()
                    batch2 = np.trim_zeros(batch2,trim='b')
                    batch2 = torch.LongTensor(batch2).unsqueeze(dim=0)
                    batch3 = batch[3][i].numpy()
                    batch3 = np.trim_zeros(batch3,trim='b')
                    batch3 = torch.LongTensor(batch3).unsqueeze(dim=0)
                    batch4 = batch[4][i].numpy()
                    batch4 = np.trim_zeros(batch4,trim='b')
                    batch4 = torch.LongTensor(batch4).unsqueeze(dim=0)
                    batch5 = batch[5][i].numpy()
                    batch5 = np.trim_zeros(batch5,trim='b')
                    batch5 = torch.LongTensor(batch5).unsqueeze(dim=0)
                    batch6 = batch[6][i].numpy()
                    batch6 = np.trim_zeros(batch6,trim='b')
                    batch6 = torch.LongTensor(batch6).unsqueeze(dim=0)
                    batch7 = batch[7][i].numpy()
                    batch7 = np.trim_zeros(batch7,trim='b')
                    batch7 = torch.LongTensor(batch7).unsqueeze(dim=0)
                    batch1 = batch[1][i].numpy()
                    batch1 = batch1[batch1 != True]
                    batch1 = torch.LongTensor(batch1).unsqueeze(dim=0)
                    preds, _, reward = re_model.predict((batch0,batch1,batch2,batch3,batch4,batch5,batch6,batch7,batch[8][i],
                                               batch[9][i].unsqueeze(dim=0), batch[10][i]), unsort=False)
                    expect += prob*reward
                else:
                    expect += prob*1.2
        print(expect.item())
        expect.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return None

    def predict(self, batch):
        sent_emb, split_idx, rels = batch[11], batch[12], batch[9]
        self.HR.eval()
        self.LR.eval()
        self.sentence_encoder.eval()
        new_batch = []
        for i in range(len(sent_emb)):
            instance_sent_emb = sent_emb[i]
            instance_sent_emb = torch.Tensor(instance_sent_emb)
            if self.opt['cuda']:
                instance_sent_emb = instance_sent_emb.cuda()
            sent_encoding = self.sentence_encoder(instance_sent_emb)
            vote_count = 0
            for sent in sent_encoding:
                prob = self.HR(torch.unsqueeze(sent, dim=0), rels[i])
                if prob>0.5:
                    vote_count += 1
                print(prob,"******")
            if float(vote_count)/len(sent_encoding) > 0.5:
                # keep the index of LR-selected sentences
                sent_idx_list, sent_idx = [], 0
                for sent_ in sent_encoding:
                    prob_ = self.LR(torch.unsqueeze(sent_, dim=0))
                    if prob_>0.5:
                        sent_idx_list.append(sent_idx)
                    sent_idx += 1
                new_batch.append(monte_carlo_sample(None, sent_idx_list, sent_idx, batch, i, False))
        return new_batch
