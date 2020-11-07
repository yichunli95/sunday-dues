import torch
import torch.nn as nn
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
    model = LinearModel()
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