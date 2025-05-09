import torch
from torch.nn.functional import softmax
from torch.autograd import Variable
from modeling.lstm.models import NLINet
import os
from utils import load_json, PROJECT_ROOT_DIR
from modeling.lstm.data import get_nli, build_vocab, get_batch
import numpy as np
from typing import Dict

class TrainedLSTMModel():

    def __init__(self, saved_dir: str):
        self.nli_net, self.word_vec, self.training_params = self.load_model(saved_dir)

    def load_model(self, saved_dir):
        config = load_json(os.path.join(saved_dir, 'infersent/config_nli_model.json'))
        training_params = load_json(os.path.join(saved_dir, 'infersent/params.json'))
        nli_net = NLINet(config).cuda()
        nli_net.load_state_dict(torch.load(
            os.path.join(PROJECT_ROOT_DIR, saved_dir, 'infersent/infersent.pth')
        ))
        nli_net.eval()
        train, valid, test = get_nli(os.path.join(PROJECT_ROOT_DIR, saved_dir))
        word_vec = build_vocab(train['s1'] + train['s2'] + valid['s1'] + valid['s2'] + test['s1'] + test['s2'], os.path.join(PROJECT_ROOT_DIR, 'modeling/lstm/dataset/GloVe/glove.840B.300d.txt'))
        return nli_net, word_vec, training_params

    def form_input(self,sent: str, word_vec: Dict[str, np.ndarray]):
        return np.array([['<s>'] + [word for word in sent.split() if word in word_vec] + ['</s>']])

    def predict(self, s1: str, s2: str):
        s1, s2 = self.form_input(s1, self.word_vec), self.form_input(s2, self.word_vec)
        s1_batch, s1_len = get_batch(s1, self.word_vec, self.training_params['word_emb_dim'])
        s2_batch, s2_len = get_batch(s2, self.word_vec, self.training_params['word_emb_dim'])
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        # model forward
        output = softmax(self.nli_net((s1_batch, s1_len), (s2_batch, s2_len)).data)

        prediction = output.data.max(1)[1].item()
        confidence = output[0].cpu().numpy()
        return prediction, confidence