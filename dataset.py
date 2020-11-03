from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
import pandas as pd

class MemexQA(Dataset):
    def __init__(self, data, shared):
        # self.data(xx_data) in original utils.py 
        self.data = data
        # self.shared(xx_shared) in original utils.py 
        self.shared = shared
    
    def __len__(self):
        return len(self.data['q'])
    
    def __getitem__(self, idx):
        # returned_item: {'q': ..., 'idxs': ..., 'aid': [0,1,2], ... (fields in xx_data.p), 
        # 'album_title': [a0_title, a1_title, a2_title], 'album_when': [a0_when, a1_when, a2_when],
        # ... (fields in xx_shared.p),}
        returned_item = {}
        
        # get fields from self.data
        #['q', 'idxs', 'cy', 'ccs', 'qid', 'y', 'aid', 'cq', 'yidx', 'cs']
        for k in self.data.keys():
            returned_item[k] = self.data[k][idx]
        
        
        # get fields from self.shared
        # dict_keys(['albums', 'pid2feat', 'word2vec', 'charCounter', 'wordCounter'])
        aid_list = returned_item['aid']
        album_data = defaultdict(list)
        for aid in aid_list:
            for key, val in self.shared['albums'][aid].items():
                album_data[key].append(val)
        returned_item.update(album_data)  
        
        # get image features  
        pid2idx = {}
        pid_list = []
        for aid in aid_list:
            pids = self.shared['albums'][aid]['photo_ids']
            for pid in pids:
                if pid not in pid_list:
                    pid_list.append(pid)
                    pid2idx[pid] = len(pid2idx.keys())
        
        image_feats = np.zeros((len(pid_list), self.shared['pid2feat'][list(self.shared['pid2feat'].keys())[0]].shape[0]), dtype="float32")
        for i in range(image_feats.shape[0]):
            image_feats[i] = self.shared['pid2feat'][pid_list[i]]
        returned_item['pidx2feat'] = image_feats
        returned_item['photo_idxs'] = [[pid2idx[pid] for pid in pids] for pids in returned_item['photo_ids']]
        return returned_item

if __name__ == '__main__':
    train_data = pd.read_pickle('prepro_v1.1/train_data.p')
    train_shared = pd.read_pickle('prepro_v1.1/train_shared.p')
    dataset = MemexQA(train_data, train_shared)
    sample = dataset[9]
    # print(sample['photo_ids'])
    # l = 0
    # for ls in sample['photo_ids']:
    #     l += len(ls)
    # print(l)
    # print(sample['photo_idxs'])
    # print(len(sample['photo_idxs']))