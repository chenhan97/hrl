import torch
import torch.nn as nn

gpu = torch.cuda.is_available()


class SentenceEncodeModule(nn.Module):
    def __init__(self):
        super(SentenceEncodeModule, self).__init__()
        self.lstm = nn.LSTM(input_size=384, hidden_size=32, bidirectional=True, batch_first=True)

    def forward(self, x):
        sents_encoding, (hn, cn) = self.lstm(x)
        return sents_encoding


class HigherReinforceModule(nn.Module):
    def __init__(self, rel_dim, hidden_dim):
        super(HigherReinforceModule, self).__init__()
        self.relation_vector = torch.nn.Parameter(torch.randn(5, rel_dim))
        self.relation_vector.requires_grad = True
        self.project_layer = nn.Sequential(nn.Linear(hidden_dim+rel_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, sent_encoding, relation_idx):
        """
        only process one sentence at a time!
        :param sent_encoding: the LSTM embedding of one sentence. format:[sentence_dim]
        :param relation_idx: the index of DS-labeled relation of this sentence(or instance)
        :return: the probability of option. format:[1], e.g., [0.99]
        """
        state = torch.cat((sent_encoding.squeeze(), self.relation_vector[relation_idx]), dim=0)
        results = torch.sigmoid(self.project_layer(state))
        return results


class LowerReinforceModule(nn.Module):
    def __init__(self, hidden_dim, ):
        super(LowerReinforceModule, self).__init__()
        self.project_layer = nn.Sequential(nn.Linear(hidden_dim, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, sent_encoding):
        """
        :param sent_encoding: the LSTM embedding of one sentence. format:[sentence_dim]
        :return: the probability of action. format:[1], e.g., [0.99]
        """
        return torch.sigmoid(self.project_layer(sent_encoding))
