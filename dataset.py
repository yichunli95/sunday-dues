import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from statistics import median

Q_THRES = int(median(q_lens))
  # 8
CS_THRES = int(median(cs_lens))  # 1
PS_THRES = int(median(photos_lens)) # 9
PTS_THRES = int(median(pts_lens)) # 3
PTS_TOTAL_THRES = PTS_THRES * PS_THRES # 27

class MemexQA(Dataset):
    def __init__(self, data, shared):
        self.data = data
        self.shared = shared
    
    def __len__(self):
        return len(self.data['q'])
    
    def __getitem__(self, idx):
        returned_item = {}
        
        # question glove
        q = self.data['q'][idx]
        q_vec = torch.FloatTensor([self.shared['word2vec'][word.lower()] for word in q if word.lower() in self.shared['word2vec']])
        if len(q_vec) < Q_THRES:
            q_vec = F.pad(q_vec, (0, 0, 0, Q_THRES - len(q_vec)))
        else:
            q_vec = q_vec[:Q_THRES]
        returned_item['q_vec'] = q_vec  # shape (8, 100)
        
        # choices glove
        wrong_cs = self.data['cs'][idx]
        correct_c = self.data['y'][idx]
        yidx = self.data['yidx'][idx]
        if yidx == 0:
            cs = [correct_c] + wrong_cs
        elif yidx == 1:
            cs = wrong_cs[:1] + [correct_c] + wrong_cs[1:]
        elif yidx == 2:
            cs = wrong_cs[:2] + [correct_c] + wrong_cs[2:]
        else:  # yidx == 3
            cs = wrong_cs + [correct_c]
        cs_vec = torch.FloatTensor()
        for c in cs:
            c_vec = torch.FloatTensor([self.shared['word2vec'][word.lower()] for word in c if word.lower() in self.shared['word2vec']])
            if c_vec.nelement() == 0:
                c_vec = torch.zeros((CS_THRES, 100))
            elif len(c_vec) < CS_THRES:
                c_vec = F.pad(c_vec, (0, 0, 0, CS_THRES - len(c_vec)))
            else:
                c_vec = c_vec[:CS_THRES]
            cs_vec = torch.cat((cs_vec, c_vec))
        returned_item['cs_vec'] = torch.unsqueeze(cs_vec, 1)  # since CS_THRES = 1 => shape (4, 1, 100)
        
        # Caveat #1: photos belong to multiple albums
        # Caveat #2: currently concat all words
        aid_list = self.data['aid'][idx]
        # photo titles glove
        pts = [pt for aid in aid_list for pt in self.shared['albums'][aid]['photo_titles']]
        pts_vec = torch.FloatTensor()
        for pt in pts:
            pt_vec = torch.FloatTensor([self.shared['word2vec'][word.lower()] for word in pt if word.lower() in self.shared['word2vec']])
            pts_vec = torch.cat((pts_vec, pt_vec))
        if pts_vec.nelement() == 0:
            pts_vec = torch.zeros((PTS_TOTAL_THRES, 100))
        elif len(pts_vec) < PTS_TOTAL_THRES:
            pts_vec = F.pad(pts_vec, (0, 0, 0, PTS_TOTAL_THRES - len(pts_vec)))
        else:
            pts_vec = pts_vec[:PTS_TOTAL_THRES]
        returned_item['pts_vec'] = pts_vec
        
        # image features of photos
        pid_list = [pid for aid in aid_list for pid in self.shared['albums'][aid]['photo_ids']]
        img_feats = torch.FloatTensor([self.shared['pid2feat'][pid] for pid in pid_list])
        if len(pid_list) < PS_THRES:
            img_feats = F.pad(img_feats, (0, 0, 0, PS_THRES - len(pid_list)))
        else:
            img_feats = img_feats[:PS_THRES]
        returned_item['img_feats'] = img_feats
        
        return returned_item, yidx



if __name__ == '__main__':
    train_data = pd.read_pickle('prepro_v1.1/train_data.p')
    train_shared = pd.read_pickle('prepro_v1.1/train_shared.p')

    data = MemexQA(train_data, train_shared)

    for i in range(len(data)):
        X, y = data[i]
        assert X['q_vec'].shape == (8, 100), X['q_vec'].shape
        assert X['cs_vec'].shape == (4, 1, 100)
        assert X['pts_vec'].shape == (27, 100)
        assert X['img_feats'].shape == (9, 2537)