import torch
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.nn.functional as F
import new_dataset_checked_by_hongyuan
import pandas as pd

class AttentionModel(nn.Module):
    def __init__(self, q_cs_input_size, desc_input_size, img_input_size, hidden_size, batch_size,
                num_layers, device, img_linear_size,num_choices = 4, rnn_type = 'bilstm'):
        super(AttentionModel, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.q_cs_input_size = q_cs_input_size   #input_size qvec and cs_vec
        self.desc_input_size = desc_input_size
        self.img_input_size = img_input_size
        self.img_linear_size = img_linear_size # s2
        self.num_directions = 2 if rnn_type == 'bilstm' else 1
        self.num_layers = num_layers
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.softmax1 = nn.Softmax(dim = 1)
        self.softmax2 = nn.Softmax(dim = 2)

        if (rnn_type == 'bilstm'):
            self.rnn_q = nn.LSTM(q_cs_input_size, hidden_size, self.num_layers, batch_first = False, bidirectional = True) # questions
            self.rnn_c = nn.LSTM(q_cs_input_size, hidden_size, self.num_layers, batch_first = False, bidirectional = True) # choices
            self.rnn_desc = nn.LSTM(desc_input_size, hidden_size, self.num_layers, batch_first = False, bidirectional = True) # photo titles
            self.rnn_ps = nn.LSTM(img_input_size, hidden_size, self.num_layers, batch_first = False, bidirectional = True) # image features
        else:
            self.rnn_q = nn.LSTM(q_cs_input_size, hidden_size, self.num_layers, batch_first = False, bidirectional = False)
            self.rnn_c = nn.LSTM(q_cs_input_size, hidden_size, self.num_layers, batch_first = False, bidirectional = False)
            self.rnn_desc = nn.LSTM(desc_input_size, hidden_size, self.num_layers, batch_first = False, bidirectional = False)
            self.rnn_ps = nn.LSTM(img_input_size, hidden_size, self.num_layers, batch_first = False, bidirectional = False)

        self.vis_text = nn.Linear(2*hidden_size, 2*hidden_size)
      
        self.tanh1 = nn.Tanh()
        self.CH_linear = nn.Linear(2*hidden_size, 2*hidden_size)
        self.img_linear = nn.Linear(hidden_size, self.img_linear_size) 
        self.tanh2 = nn.Tanh()
        self.tanh3 = nn.Tanh()

        #input: B X num_choices X (5 * 2 * hidden_size), output: B x num_choices X 1
        self.last_softmax = nn.Linear(5 * 2 * hidden_size , 1) 

    
    def forward(self, X):
        # X is a list of dictionaries: 'q_vec', 'cs_vec', 'desc_vec', 'img_feats'
        # BATCH_SIZE = len(X)
        # B, T, F
        # q_vec -> B, T, 100 (glove embedding)
        # cs_vec -> B, 4, Y_THRES, 100
        # desc_vec(prev. pt) -> B, all_photo_titles_albums * 40, 100
        # img_feats -> B, all_photo_titles_albums, 2537

        q_vec = X['q_vec']
        cs_vec = X['cs_vec']
        desc_vec = X['desc_vec']
        img_feats = X['img_feats']

        packed_q_vec = pack_padded_sequence(q_vec, X['q_len'], batch_first=False, enforce_sorted = False)
        packed_c0_vec = pack_padded_sequence(cs_vec[:, :, 0, :], X['cs0_lens'], batch_first = False, enforce_sorted = False)
        packed_c1_vec = pack_padded_sequence(cs_vec[:, :, 1, :], X['cs1_lens'], batch_first = False, enforce_sorted = False)
        packed_c2_vec = pack_padded_sequence(cs_vec[:, :, 2, :], X['cs2_lens'], batch_first = False, enforce_sorted = False)
        packed_c3_vec = pack_padded_sequence(cs_vec[:, :, 3, :], X['cs3_lens'], batch_first = False, enforce_sorted = False)
        packed_pt_vec = pack_padded_sequence(desc_vec, X['desc_len'], batch_first = False, enforce_sorted = False)
        packed_img_vec = pack_padded_sequence(img_feats, X['img_len'], batch_first = False, enforce_sorted = False)
        q_out,_   = self.rnn_q(packed_q_vec)  # M X B X 2d
        c0_out,_  = self.rnn_c(packed_c0_vec) # T X B X 2d
        c1_out,_  = self.rnn_c(packed_c1_vec) # T X B X 2d
        c2_out,_  = self.rnn_c(packed_c2_vec) # T X B X 2d
        c3_out,_  = self.rnn_c(packed_c3_vec) # T X B X 2d
        text_out,_ = self.rnn_desc(packed_pt_vec) # T X B X 2d
        vis_out,_  = self.rnn_ps(packed_img_vec) # T X B X 2d

        print("vis_out: ", vis_out.shape, text_out.shape)


        q_out, q_lens_unpacked = pad_packed_sequence(q_out, batch_first=False)  # M X B X 2d
        q_out = q_out.permute(1,0,2) # B X M X 2d
        c0_out, c0_lens_unpacked = pad_packed_sequence(c0_out, batch_first=False)  # T X B X 2d
        c0_out = c0_out[-1,:,:].unsqueeze(0).permute(1,0,2) # B X 1 X 2d
        c1_out, c1_lens_unpacked = pad_packed_sequence(c1_out, batch_first=False)  # T X B X 2d
        c1_out = c1_out[-1,:,:].unsqueeze(0).permute(1,0,2) # B X 1 X 2d
        c2_out, c2_lens_unpacked = pad_packed_sequence(c2_out, batch_first=False)    # T X B X 2d
        c2_out = c2_out[-1,:,:].unsqueeze(0).permute(1,0,2) # B X 1 X 2d
        c3_out, c3_lens_unpacked = pad_packed_sequence(c3_out, batch_first=False)   # T X B X 2d
        c3_out = c3_out[-1,:,:].unsqueeze(0).permute(1,0,2) # B X 1 X 2d
        txt_out, text_out_lens = pad_packed_sequence(text_out, batch_first=False) # T X B X 2d
        txt_out = txt_out.permute(1,0,2) # B X T X 2d
       

        vis_out, vis_out_lens = pad_packed_sequence(vis_out, batch_first=False) # T X B X 2d
        vis_out = vis_out.permute(1,0,2) # B X T X 2d    
        
        # print("vis_out: ", vis_out.shape)

        # print("qout: ", q_out.shape)
        # correlation b/w txt_out and vis_out : B X T X 2d -> linear_layer 
        vis = self.vis_text(vis_out)      # B X T X 2d
        vis_out = torch.unsqueeze(vis_out, 3) 
        # print("vis: ", vis.shape)
        text = self.vis_text(txt_out)      # B X T X 2d
        txt_out = torch.unsqueeze(txt_out, 3)
       
        H = torch.cat([vis_out, txt_out], dim = 3)  # B * T * 2d * 2
        # print("H: ", H.shape)  
        C = self.tanh1(vis @ text.permute(0,2,1))  # B X T X T
        C_repeat = C.unsqueeze(3).repeat(1,1,1,2) # B X T x T X 2
        # print("C: ", C_repeat.shape)
        
        
        H_perm = H.permute(0, 3, 2, 1) # B x 2 x 2d x T
        # print("Hperm shape:", H_perm.shape)
        C_perm = C_repeat.permute(0,3,1,2) # B x 2 x T x T
        # print("Cperm shape:", C_perm.shape)
        F = torch.matmul(H_perm, C_perm) # B x 2 X 2d x T
       
        F = F.permute(0,3,1,2) #F:  B * T * 2 * 2d
        F = self.CH_linear(F) 
        F = self.tanh2(F)
        # print("F shape:", F.shape) 

        E = torch.cat([c0_out,c1_out, c2_out, c3_out], dim = 1)  #B x 4 X 2d
        # print("E shape:", E.shape) 
        Q = q_out  #   Q: B x M x 2d
        # print("Q shape:", Q.shape) 
        
        bmm1 = Q.bmm(F.permute(0,3,1,2)[:,:,:,0]).unsqueeze(3)
        bmm2 = Q.bmm(F.permute(0,3,1,2)[:,:,:,1]).unsqueeze(3)
        S = torch.cat([bmm1, bmm2], dim = 3) #S:  B x M x T x 2
        # print("S shape:", S.shape) 
        S = self.tanh3(S)
        max1 = torch.max(S, dim = 1)[0]
        # print("max1 shape: ", max1.shape)
        max2 = torch.max(max1, dim = 1)[0]
        # print("max2 shape: ", max2.shape)
        max3 = torch.max(S, dim = 3)[0]
        # print("max3 shape: ", max3.shape)
        max4 = torch.max(max3, dim = 2)[0]
        # print("max4 shape: ", max4.shape)

        '''
        max1 shape:  torch.Size([3, 69, 2])
        max2 shape:  torch.Size([3, 2])
        max4 shape:  torch.Size([3, 9, 2])
        max3 shape:  torch.Size([3, 9])
                '''
        A = self.softmax1(max1)  # B X T x 2
        B = self.softmax1(max2)   # B X 2
        D = self.softmax1(max4)   # B x M
      
        h_tilda = torch.zeros((F.shape[0], F.shape[-1])).float() # B x 2d
        q_tilda = torch.zeros((F.shape[0], F.shape[-1])).float() # B x 2d
       
        for k in range(2):
            temp = torch.zeros((F.shape[0], F.shape[-1])).float()
            for t in range(A.shape[1]):
                # print("A shape: ", A[:,t, k].unsqueeze(1).repeat(1, F.shape[-1]).shape, " , F[:, t, k, :] shape: ",  F[:, t, k, :].shape )
                temp += A[:,t, k].unsqueeze(1).repeat(1, F.shape[-1]) *  F[:, t, k, :]# B x 2d
                # print("temp shape: ", temp.shape)
            h_tilda +=  (B[:, k].unsqueeze(1).repeat(1, F.shape[-1]) * temp)
            # print("htilda shape: ", h_tilda.shape)

        
        H_tilda = h_tilda.unsqueeze(1).repeat(1, 4, 1)  # B x 4 X 2d
        # print("htilda: ", H_tilda.shape)

        for m in range(D.shape[1]):
            # print("D shape: ",D[:, m].unsqueeze(1).repeat(1, Q.shape[-1]).shape, " Q shape: ", Q[:, m, :].shape )
            q_tilda += D[:, m].unsqueeze(1).repeat(1, Q.shape[-1]) * Q[:, m, :]
            # print("qtilda: ", q_tilda.shape)

        Q_tilda = q_tilda.unsqueeze(1).repeat(1, 4, 1)  # B x 4 X 2d
        # print("qtilda: ", Q_tilda.shape)
        concat = torch.cat([Q_tilda, H_tilda, E, Q_tilda * E, H_tilda * E], dim = -1)

        output = self.last_softmax(concat)

        output = output.squeeze(2)
        out_softmax = self.softmax1(output)
        return out_softmax
