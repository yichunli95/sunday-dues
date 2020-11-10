import torch
import torch.nn as nn
from torch.nn.utils.rnn import *
import dataset
import pandas as pd

class LinearModel(nn.Module):
    def __init__(self, device):
        super(LinearModel, self).__init__()
        self.device = device
        self.img_feats_reshape = nn.Linear(2537, 100)
        self.flat = nn.Flatten()
        self.output = nn.Linear((dataset.Q_THRES + dataset.CS_THRES + \
            dataset.PTS_TOTAL_THRES + dataset.PS_THRES) * 100, 1)

    def forward(self, X):
        # X is a list of dictionaries: 'q_vec', 'cs_vec', 'pts_vec', 'img_feats'
        # BATCH_SIZE = len(X)
        img_feats = X['img_feats'].to(self.device)
        img_feats = self.img_feats_reshape(img_feats) # image_feats (BATCH_SIZE, 9, 2537) -> (BATCH_SIZE, 9, 100) 
        q_vec = X['q_vec'].to(self.device)  # question (BATCH_SIZE, 8, 100)
        cs_vec = X['cs_vec'].to(self.device) # 4 choices (BATCH_SIZE, 4, 1, 100)
        pts_vec = X['pts_vec'].to(self.device) # photo titles (BATCH_SIZE, 27, 100)
        #print('Q len: ', X['q_len'])
        #print('CS lens: ', X['cs_lens'])
        #print('Pts len: ', X['pts_len'])
        # for item in X:
        #     q_vec = torch.cat((q_vec, torch.unsqueeze(item['q_vec'], 0)), dim = 0)
        #     cs_vec = torch.cat((cs_vec, torch.unsqueeze(item['cs_vec'], 0)), dim = 0)
        #     pts_vec = torch.cat((pts_vec, torch.unsqueeze(item['pts_vec'], 0)), dim = 0)

        #     # reshape img_feats and concat
        #     img_reshape = self.img_feats_reshape(item['img_feats'])
        #     img_feats = torch.cat((img_feats, torch.unsqueeze(img_reshape, 0)), dim = 0)

        concat_vec = torch.FloatTensor().to(self.device)
        for i in range(4):
            vec_to_be_cat = torch.unsqueeze(self.flat(torch.cat((q_vec, img_feats, cs_vec[:, i, :], pts_vec), dim = 1)), 1)
            concat_vec = torch.cat((concat_vec, vec_to_be_cat), dim = 1)
            #concat_vec = torch.cat((concat_vec, torch.cat((q_vec, img_feats, cs_vec[:, i, :], pts_vec), dim = 1)), dim = 1)
        assert concat_vec.shape == torch.Size((len(X['q_vec']), 4, (dataset.Q_THRES + dataset.CS_THRES + \
            dataset.PTS_TOTAL_THRES + dataset.PS_THRES) * 100)), concat_vec.shape
        # len(X), 4, (dataset.Q_THRES + dataset.CS_THRES + dataset.PTS_TOTAL_THRES + dataset.PS_THRES)* 100
        out = self.output(concat_vec)
        assert out.shape == torch.Size((len(X['q_vec']), 4, 1)), out.shape
        out = out.squeeze(2)
        assert out.shape == torch.Size((len(X['q_vec']), 4))
        return out


class SimpleLSTMModel(nn.Module):
    '''
    def __init__(self, device):
        super(SimpleLSTMModel, self).__init__()
        self.device = device
        self.img_feats_reshape = nn.Linear(2537, 100)
        self.flat = nn.Flatten()
        self.output = nn.Linear((dataset.Q_THRES + dataset.CS_THRES + \
            dataset.PTS_TOTAL_THRES + dataset.PS_THRES) * 100, 1)
    '''

    def __init__(self, input_size, hidden_size, batch_size, num_layers, device, rnn_type = 'bilstm'):
        super(SimpleLSTMModel, self).__init__()
        self.device = device
        self.img_feats_reshape = nn.Linear(2537, 100)
        self.flat = nn.Flatten()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size
        if (rnn_type == 'bilstm'):
            self.rnn_q = nn.LSTM(input_size, hidden_size, num_layers, batch_first = False, bidirectional = True)
            self.rnn_c = nn.LSTM(input_size, hidden_size, num_layers, batch_first = False, bidirectional = True)
            self.rnn_pt = nn.LSTM(input_size, hidden_size, num_layers, batch_first = False, bidirectional = True)
            self.rnn_ps = nn.LSTM(input_size, hidden_size, num_layers, batch_first = False, bidirectional = True)
        self.output = nn.Linear(4 * num_layers * (2 if rnn_type == 'bilstm' else 1), 1)  # 4 * hidden size is the concatenated dim
            

    def forward(self, X):
        # X is a list of dictionaries: 'q_vec', 'cs_vec', 'pts_vec', 'img_feats'
        # BATCH_SIZE = len(X)
        img_feats = X['img_feats'].to(self.device)
        img_feats = self.img_feats_reshape(img_feats) # image_feats (BATCH_SIZE, 9, 2537) -> (BATCH_SIZE, 9, 100) 
        q_vec = X['q_vec'].to(self.device)  # question (BATCH_SIZE, 8, 100)
        cs_vec = X['cs_vec'].to(self.device) # 4 choices (BATCH_SIZE, 4, 1, 100)
        pts_vec = X['pts_vec'].to(self.device) # photo titles (BATCH_SIZE, 27, 100)
        packed_q_vec = pack_padded_sequence(q_vec, X['q_len'][0], batch_first = True, enforce_sorted = False)
        packed_c1_vec = pack_padded_sequence(cs_vec[:, 0, :, :], X['cs_lens'][0], batch_first = True, enforce_sorted = False)
        packed_c2_vec = pack_padded_sequence(cs_vec[:, 1, :, :], X['cs_lens'][1], batch_first = True, enforce_sorted = False)
        packed_c3_vec = pack_padded_sequence(cs_vec[:, 2, :, :], X['cs_lens'][2], batch_first = True, enforce_sorted = False)
        packed_c4_vec = pack_padded_sequence(cs_vec[:, 3, :, :], X['cs_lens'][3], batch_first = True, enforce_sorted = False)
        packed_pt_vec = pack_padded_sequence(pts_vec, X['pts_len'][0], batch_first = True, enforce_sorted = False)
        img_feats = img_feats.permute(1,0,2)  # batch_size, seq_len, input_size -> seq_len, batch_size, input_size
        _, (lstm_output_q, __) = self.rnn_q(packed_q_vec)
        _, (lstm_output_c1, __) = self.rnn_c(packed_c1_vec)
        _, (lstm_output_c2, __) = self.rnn_c(packed_c2_vec)
        _, (lstm_output_c3, __) = self.rnn_c(packed_c3_vec)
        _, (lstm_output_c4, __) = self.rnn_c(packed_c4_vec)
        _, (lstm_output_pt, __) = self.rnn_pt(packed_pt_vec)
        _, (lstm_output_ps, __) = self.rnn_ps(img_feats) # 4, 10, 1
        lstm_output_cs = [lstm_output_c1, lstm_output_c2, lstm_output_c3, lstm_output_c4]
      
        concat_vec = torch.FloatTensor().to(self.device)
        for i in range(4):
            vec_to_be_cat = torch.cat((lstm_output_q, lstm_output_ps, lstm_output_cs[i], lstm_output_pt), dim = 0) # 16, 10, 1
            print("vec_to_be_cat shape: ", vec_to_be_cat.shape)
            concat_vec = torch.cat((concat_vec, vec_to_be_cat), dim = 2) # 16, 10, 4
            print("concat_vec: ", concat_vec.shape)
        concat_vec = concat_vec.permute(1, 2, 0)  # 10, 4, 16
        # for item in X:
        #     q_vec = torch.cat((q_vec, torch.unsqueeze(item['q_vec'], 0)), dim = 0)
        #     cs_vec = torch.cat((cs_vec, torch.unsqueeze(item['cs_vec'], 0)), dim = 0)
        #     pts_vec = torch.cat((pts_vec, torch.unsqueeze(item['pts_vec'], 0)), dim = 0)

        #     # reshape img_feats and concat
        #     img_reshape = self.img_feats_reshape(item['img_feats'])
        #     img_feats = torch.cat((img_feats, torch.unsqueeze(img_reshape, 0)), dim = 0)
        '''
        concat_vec = torch.FloatTensor().to(self.device)
        for i in range(4):
            vec_to_be_cat = torch.unsqueeze(self.flat(torch.cat((q_vec, img_feats, cs_vec[:, i, :], pts_vec), dim = 1)), 1)
            concat_vec = torch.cat((concat_vec, vec_to_be_cat), dim = 1)
            #concat_vec = torch.cat((concat_vec, torch.cat((q_vec, img_feats, cs_vec[:, i, :], pts_vec), dim = 1)), dim = 1)
        assert concat_vec.shape == torch.Size((len(X['q_vec']), 4, (dataset.Q_THRES + dataset.CS_THRES + \
            dataset.PTS_TOTAL_THRES + dataset.PS_THRES) * 100)), concat_vec.shape
        '''
        # len(X), 4, (dataset.Q_THRES + dataset.CS_THRES + dataset.PTS_TOTAL_THRES + dataset.PS_THRES)* 100
        print("out before output layer: ", concat_vec.shape)
        out = self.output(concat_vec)
        #assert out.shape == torch.Size((len(X['q_vec']), 4, 1)), out.shape
        out = out.squeeze(2)
        print("out size ", out.shape)
        #assert out.shape == torch.Size((len(X['q_vec']), 4))
        return out


if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = SimpleLSTMModel(100, 1, 10, 2, device)
    train_data = pd.read_pickle('prepro_v1.1/train_data.p')
    train_shared = pd.read_pickle('prepro_v1.1/train_shared.p')
    data = dataset.MemexQA_simple(train_data, train_shared)
    print(len(data))
    loader = torch.utils.data.DataLoader(data, batch_size = 10)
    for X, y in loader:
        print(type(X))
        # print(X.keys())
        # print(len(X['q_vec']))
        # print(X['q_vec'][0].shape)
        print(model(X))
        break