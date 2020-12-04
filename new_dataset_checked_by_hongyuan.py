import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import itertools

train_data = pd.read_pickle('prepro_v1.1/train_data.p')
train_shared = pd.read_pickle('prepro_v1.1/train_shared.p')

q_lens = [len(q) for q in train_data['q']]
cs_lens = [[len(c) for c in cs] for cs in train_data['cs']]
cs_lens = list(itertools.chain(*cs_lens))
y_lens = [len(y) for y in train_data['y']]
photo_lens = [len(train_shared['albums'][aid]['photo_ids']) for aid in train_shared['albums']]
all_photos_lens = [sum(len(train_shared['albums'][aid]['photo_ids']) for aid in aid_list) for aid_list in train_data['aid']]
pts_lens = [len(pt) for aid in train_shared['albums'] for pt in train_shared['albums'][aid]['photo_titles']] #number of photos/album
when_lens = [len(train_shared['albums'][aid]['when']) for aid in train_shared['albums']]
album_title_lens = [len(train_shared['albums'][aid]['title']) for aid in train_shared['albums']]
album_desc_lens = [len(train_shared['albums'][aid]['description']) for aid in train_shared['albums']]

Q_THRES = int(np.percentile(q_lens, 90)) # 10
Y_THRES = int(np.percentile(cs_lens, 90)) # 3, same as np.percentile(y_lens, 90)
PTS_THRES = int(np.percentile(pts_lens, 90)) # 8
WHEN_THRES = int(np.percentile(when_lens, 90)) # 4
PHOTOS_PER_ALBUM = int(np.percentile(photo_lens, 90)) # 10
ALBUM_TITLE_THRES = int(np.percentile(album_title_lens, 90)) # 8
ALBUM_DESC_THRES = int(np.percentile(album_desc_lens, 50)) # 11


class MemexQA_new(Dataset):
    def __init__(self, data, shared):
        self.data = data
        self.shared = shared

    def __len__(self):
        return len(self.data['q'])

    def __getitem__(self, idx):
        returned_item = {}
        # self.data keys -> ['q', 'idxs', 'cy', 'ccs', 'qid', 'y', 'aid', 'cq', 'yidx', 'cs']
        # self.shared keys -> ['albums', 'pid2feat', 'word2vec', 'charCounter', 'wordCounter']

        q = self.data['q'][idx]
        # missing glove word-> [0] * 100 embedding
        q_vec = torch.FloatTensor(
            [self.shared['word2vec'][word.lower()] if word.lower() in self.shared['word2vec'] else [0] * 100 for word in
             q])
        returned_item['q_vec'] = q_vec[:Q_THRES]  # largest possible shape: Q_THRES * 100

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

        cs_vec = [
            [self.shared['word2vec'][word.lower()] if word.lower() in self.shared['word2vec'] else [0] * 100 for word in
             c] for c in cs]
        cs_vec = [torch.FloatTensor(c[:Y_THRES]) for c in cs_vec]
        returned_item['cs_vec'] = cs_vec  # [c1, c2, c3, c4]; largest possible shape: 4, Y_THRES, 100

        # aid: description + title , aid:when , aid : photo_titles + {later ->( photo_captions  + photo tags )}
        aid_list = self.data['aid'][idx]
        pts_descs = []  # photo-level
        pid_features = []  # img features from pre-trained CNN
        # for each album
        total_cat_len = ALBUM_TITLE_THRES + ALBUM_DESC_THRES + WHEN_THRES + PTS_THRES  # 8 + 11 + 4 + 8 = 31
        for aid in aid_list:
            # ptags = {for each self.album_itags[aid]}
            album = self.shared['albums'][aid]
            pts = album['photo_titles']  # all photo titles/aid

            # concatenate album description, album title and album when
            desc = album['description'][:ALBUM_DESC_THRES] + album['title'][:ALBUM_TITLE_THRES] + album['when'][
                                                                                                  :WHEN_THRES]

            for pt in pts:
                photo_info = desc + pt[:PTS_THRES]
                # largest possible shape: total_cat_len, 100
                photo_info_vec = [
                    self.shared['word2vec'][word.lower()] if word.lower() in self.shared['word2vec'] else [0] * 100 for
                    word in photo_info]
                if len(photo_info_vec) < total_cat_len:
                    photo_info_vec = photo_info_vec + [[0] * 100 for _ in range(
                        total_cat_len - len(photo_info_vec))]  # total_cat_len, 100
                pts_descs.append(photo_info_vec)  # total number of photos (varies), total_cat_len, 100

            for pid in self.shared['albums'][aid]['photo_ids']:
                # img_feats
                pid_features.append(self.shared['pid2feat'][pid])  # total number of photos (varies) * 2537

        desc_vec = torch.FloatTensor(pts_descs).view(-1,
                                                     total_cat_len * 100)  # total number of photos (varies), total_cat_len * 100
        returned_item['desc_vec'] = desc_vec

        img_feats_vec = torch.FloatTensor(
            pid_features)  # total number of photos (varies), 2537; NEWLY CHANGED (no matter what, it will vary; keep consistent with desc_vec)
        returned_item['img_feats'] = img_feats_vec
        return returned_item, yidx