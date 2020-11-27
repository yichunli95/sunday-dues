import torch
import torch.nn as nn
from torch.nn.utils.rnn import *
import dataset
import pandas as pd

# replace the following in SimpleLSTM:
# - img_feats_reshape (nn.Linear) is replaced with a dynamic parameter layer whose weights
# are determined based on question features (https://arxiv.org/abs/1511.05756)
# - concat_vector is replaced with a fusion CNN layer (https://arxiv.org/abs/1511.05756)
class NewFusionModel(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers, device, q_linear_size, img_linear_size, multimodal_out, kernel, stride, rnn_type = 'bilstm'):
        super(NewFusionModel, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size         
        self.q_linear_size = q_linear_size # s1
        self.img_linear_size = img_linear_size # s2
        self.num_directions = 2 if rnn_type == 'bilstm' else 1
        self.multimodal_out = multimodal_out
        self.kernel = kernel
        self.stride = stride
        self.num_layers = num_layers

        if (rnn_type == 'bilstm'):
            self.rnn_q = nn.LSTM(input_size, hidden_size, self.num_layers, batch_first = False, bidirectional = True)
            self.rnn_c = nn.LSTM(input_size, hidden_size, self.num_layers, batch_first = False, bidirectional = True)
            self.rnn_pt = nn.LSTM(input_size, hidden_size, self.num_layers, batch_first = False, bidirectional = True)
            self.rnn_ps = nn.LSTM(2537, hidden_size, self.num_layers, batch_first = False, bidirectional = True)
        else:
            self.rnn_q = nn.LSTM(input_size, hidden_size, self.num_layers, batch_first = False, bidirectional = False)
            self.rnn_c = nn.LSTM(input_size, hidden_size, self.num_layers, batch_first = False, bidirectional = False)
            self.rnn_pt = nn.LSTM(input_size, hidden_size, self.num_layers, batch_first = False, bidirectional = False)
            self.rnn_ps = nn.LSTM(2537, hidden_size, self.num_layers, batch_first = False, bidirectional = False)

        self.img_linear = nn.Linear(hidden_size, self.img_linear_size) 
        self.q_linear = nn.Linear(hidden_size, self.q_linear_size)  

        self.multimodal_cnn = nn.Conv1d(self.num_directions * self.num_layers, self.multimodal_out, self.kernel, self.stride)
        # 2 * hidden_size if not passing in rnn_q hidden output, 3 * hidden_size if passing in 
        multimodal_cnn_in_size = 2 * hidden_size + self.num_layers * self.num_directions * batch_size * q_linear_size // self.img_linear_size
        multimodal_cnn_out_size = (multimodal_cnn_in_size - self.kernel) // self.stride + 1
        self.output = nn.Linear(self.multimodal_out * multimodal_cnn_out_size, 1)
    
    def forward(self, X):
        # X is a list of dictionaries: 'q_vec', 'cs_vec', 'pts_vec', 'img_feats'
        # BATCH_SIZE = len(X)
        img_feats = X['img_feats'].to(self.device)
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
        
        _, (lstm_hidden_q, __) = self.rnn_q(packed_q_vec)
        _, (lstm_hidden_c1, __) = self.rnn_c(packed_c1_vec)
        _, (lstm_hidden_c2, __) = self.rnn_c(packed_c2_vec)
        _, (lstm_hidden_c3, __) = self.rnn_c(packed_c3_vec)
        _, (lstm_hidden_c4, __) = self.rnn_c(packed_c4_vec)
        _, (lstm_hidden_pt, __) = self.rnn_pt(packed_pt_vec)
        _, (lstm_hidden_ps, __) = self.rnn_ps(img_feats) # 4, 10, 1
        lstm_hidden_cs = [lstm_hidden_c1, lstm_hidden_c2, lstm_hidden_c3, lstm_hidden_c4]
        
        candidate_weights = self.q_linear(lstm_hidden_q) # output: (num_direction * num_layers, batch_size, self.q_linear_size)
        img_feats = self.img_linear(lstm_hidden_ps) # output: (num_direction * num_layers, batch_size, hidden_size)
        # dyanmic parameter layer
        dynamic_parameter_out = self.num_directions * self.num_layers * self.batch_size * self.q_linear_size // self.img_linear_size
        dynamic_parameter_matrix = torch.flatten(candidate_weights)[:self.img_linear_size * dynamic_parameter_out]
        dynamic_parameter_matrix = dynamic_parameter_matrix.reshape(self.img_linear_size, dynamic_parameter_out)
        q_img_fused = img_feats @ dynamic_parameter_matrix

        # multimodal cnn layer
        cnn_out_list = []
        for i in range(4):
            vec = torch.cat((q_img_fused, lstm_hidden_cs[i], lstm_hidden_pt), dim = 2).to(self.device) 
            vec = vec.permute(1, 0, 2) # batch_size, num_direction * num_layers, 2 * hidden + dynamic_parameter_out
            vec = self.multimodal_cnn(vec)
            cnn_out_list.append(vec)
        for i in range(4):
            cnn_out_list[i] = torch.flatten(cnn_out_list[i], start_dim = 1).unsqueeze(1)
        classification_input = torch.cat(cnn_out_list, dim = 1) # (batch_size, 4, out_ch * cnn_out)
        logits = self.output(classification_input)
        return logits.squeeze(2)



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
        self.output = nn.Linear(4 * num_layers * (2 if rnn_type == 'bilstm' else 1) * hidden_size, 1)  # 4 * hidden size is the concatenated dim
            

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
        _, (lstm_hidden_q, __) = self.rnn_q(packed_q_vec)
        _, (lstm_hidden_c1, __) = self.rnn_c(packed_c1_vec)
        _, (lstm_hidden_c2, __) = self.rnn_c(packed_c2_vec)
        _, (lstm_hidden_c3, __) = self.rnn_c(packed_c3_vec)
        _, (lstm_hidden_c4, __) = self.rnn_c(packed_c4_vec)
        _, (lstm_hidden_pt, __) = self.rnn_pt(packed_pt_vec)
        _, (lstm_hidden_ps, __) = self.rnn_ps(img_feats) # 4, 10, 1
        lstm_hidden_cs = [lstm_hidden_c1, lstm_hidden_c2, lstm_hidden_c3, lstm_hidden_c4]
      
        concat_vec = torch.FloatTensor().to(self.device)
        for i in range(4):
            vec_to_be_cat = torch.cat((lstm_hidden_q, lstm_hidden_ps, lstm_hidden_cs[i], lstm_hidden_pt), dim = 0) # 16, 10, 1
            vec_to_be_cat = vec_to_be_cat.permute(1, 0, 2)
            vec_to_be_cat = torch.flatten(vec_to_be_cat, 1, 2)
            vec_to_be_cat = torch.unsqueeze(vec_to_be_cat, 1)
            # we want concat_vec to have a final shape of N, 4, *
#             print("vec_to_be_cat shape: ", vec_to_be_cat.shape)
            concat_vec = torch.cat((concat_vec, vec_to_be_cat), dim = 1) # 16, 10, 4
#             print("vectobecat shape: ", vec_to_be_cat.shape)
#         print("concat_vec: ", concat_vec.shape)
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
#         print("out before output layer: ", concat_vec.shape)
        #print("concat_vec: ", concat_vec.shape) 
        out = self.output(concat_vec)
        #assert out.shape == torch.Size((len(X['q_vec']), 4, 1)), out.shape
        out = out.squeeze(2)
#         print("out size ", out.shape)
        #assert out.shape == torch.Size((len(X['q_vec']), 4))
        return out

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

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    #model = SimpleLSTMModel(100, 64, 7, 2, device)
    # input_size, hidden_size, batch_size, num_layers, device, q_linear_size, img_linear_size, multimodal_out, kernel, stride
    model = NewFusionModel(100, 64, 64, 2, device, 64, 64, 4, 3, 1)
    train_data = pd.read_pickle('prepro_v1.1/train_data.p')
    train_shared = pd.read_pickle('prepro_v1.1/train_shared.p')
    data = dataset.MemexQA_simple(train_data, train_shared)
#     print(len(data))
    loader = torch.utils.data.DataLoader(data, batch_size = 64)
    for X, y in loader:
#         print(type(X))
        # print(X.keys())
        # print(len(X['q_vec']))
        # print(X['q_vec'][0].shape)
#         print("output size: ", model(X).shape)
        break
