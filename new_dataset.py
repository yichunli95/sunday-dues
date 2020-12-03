from functools import total_ordering
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from statistics import median
import itertools
import matplotlib.pyplot as plt
import json

train_data = pd.read_pickle('prepro_v1.1/train_data.p')
train_shared = pd.read_pickle('prepro_v1.1/train_shared.p')
with open('album_info.json') as json_file: 
    album_info = json.load(json_file) 
q_lens = [len(q) for q in train_data['q']]
cs_lens = [[len(c) for c in cs] for cs in train_data['cs']]
cs_lens = list(itertools.chain(*cs_lens))
y_len = [len(yy) for yy in train_data['y']]
photos_lens = [sum(len(train_shared['albums'][aid]['photo_titles']) for aid in aid_list) for aid_list in train_data['aid']]
pts_lens = [len(pt) for aid in train_shared['albums'] for pt in train_shared['albums'][aid]['photo_titles']] #number of photos/album
desc_len = [len(pt) for aid in train_shared['albums'] for pt in train_shared['albums'][aid]['description']]
when_len = [len(pt) for aid in train_shared['albums'] for pt in train_shared['albums'][aid]['when']]
ptts_lens = [ len(pt) for aid in train_shared['albums'] for each in train_shared['albums'][aid]['photo_titles'] for pt in each]

Q_THRES = int(max(q_lens))  #23
C_THRES = int(max(cs_lens)) #11
Y_THRES = 8  #1 - 18
PTS_THRES = 12 # 
WHEN_THRES = 4
PHOTOS_PER_ALBUM = 3

class MemexQA_new(Dataset):
    def __init__(self, data, shared, info):
        self.data = data
        self.shared = shared
        self.album_info = info
        self.album_itags = {aid['album_id'] : aid['photo_tags'] for aid in self.album_info}
    
    def __len__(self):
        return len(self.data['q'])
    
    def __getitem__(self, idx):
        returned_item = {}
        # self.data keys -> ['q', 'idxs', 'cy', 'ccs', 'qid', 'y', 'aid', 'cq', 'yidx', 'cs']
        # self.shared keys -> ['albums', 'pid2feat', 'word2vec', 'charCounter', 'wordCounter']

        q = self.data['q'][idx]
        # missing glove word-> [0] * 100 embedding
        q_vec = torch.FloatTensor([self.shared['word2vec'][word.lower()] if word.lower() in self.shared['word2vec'] else [0]*100  for word in q ])
        returned_item['q_vec'] = q_vec
        returned_item['q_len'] = q_vec.shape[0]  # q_vec -> W x 100 (glove embedding)

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
        cs_vec = [[self.shared['word2vec'][word.lower()] if word.lower() in self.shared['word2vec'] else [0]*100  for word in each] for each in cs]
        cs_lens = [min(Y_THRES, len(each)) for each in cs_vec] #YTHRES
             
        cs_vec = padLength(cs_vec, cs_lens, Y_THRES)
        returned_item['cs_vec']  = torch.FloatTensor(cs_vec)  # 4 x Y_THRES X 100
        returned_item['cs_lens'] = cs_lens

        # aid: description + title , aid:when , aid : photo_titles + {later ->( photo_captions  + photo tags )}
        aid_list = self.data['aid'][idx]
        pts_descs = []
        pid_features = []
        # for each album
        total_cat_len = 3 * PTS_THRES + WHEN_THRES # PTS_THRES(album description) + PTS_THRES(album title) + WHEN_THRES(album when) + PTS_THRES(photo title) = 3 * 12 + 4 = 40
        for aid in aid_list:
            # ptags = {for each self.album_itags[aid]}
            album = self.shared['albums'][aid]
            pts = album['photo_titles']   #all photo titles/aid

            # concatenate album description, album title and album when
            desc = album['description'][:PTS_THRES] + album['title'][:PTS_THRES] + album['when'][:WHEN_THRES]
            
            for pt in pts:
                desc = desc + pt[:PTS_THRES]
                pts_vec = [self.shared['word2vec'][word.lower()] if word.lower() in self.shared['word2vec'] else [0]*100 for word in desc]
                if len(pts_vec) < total_cat_len:
                    pts_vec = pts_vec + [[0] * 100 for _ in range(total_cat_len - len(pts_vec))]  # total_cat_length X 100   
                pts_descs.append(pts_vec)
            for pid in self.shared['albums'][aid]['photo_ids'][:PHOTOS_PER_ALBUM]:
                # img_feats
                pid_features.append(self.shared['pid2feat'][pid])

        desc_vec = torch.FloatTensor(pts_descs).view(-1, total_cat_len * 100) # [all_photo_titles_albums X total_cat_length] X 100  
        returned_item['desc_vec'] = desc_vec

        img_feats_vec = torch.FloatTensor(pid_features) # ([num_of_albums * PHOTOS_PER_ALBUM], 2537)
        returned_item['img_feats'] = img_feats_vec        
        return returned_item, yidx


#pad sentences to equal lengths embedding
def padLength(vec, lens, thres):
    max_len = thres 
    for i in range(len(lens)):
        if len(vec[i]) < max_len:
            for j in range(max_len - lens[i]):  #padding of 0s of size -> [lens[i] - max_len] X 100
                vec[i].append(([0]*100)) 

    return vec
