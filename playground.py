import pickle
import pandas as pd

# display first few results of each of the fields in the pre-processed pickle files
def display_top_train_data(top=3):
    # train_shared => dict_keys(['albums', 'pid2feat', 'word2vec', 'charCounter', 'wordCounter'])
    train_shared = pd.read_pickle('prepro_v1.1/train_shared.p')
    print("train_shared.p")
    print(train_shared.keys())
    for k1 in train_shared.keys():
        print("======")
        print(k1)
        if isinstance(train_shared[k1], dict):
            for kk1 in list(train_shared[k1].keys())[:top]:
                print(kk1, train_shared[k1][kk1])
        else:
            print(train_shared[k1].most_common(top))

    train_data = pd.read_pickle('prepro_v1.1/train_data.p')
    # train_data => dict_keys(['q', 'idxs', 'cy', 'ccs', 'qid', 'y', 'aid', 'cq', 'yidx', 'cs'])
    print("======================")
    print("train_data.p")
    print(train_data.keys())
    for k2 in train_data.keys():
        print("======")
        print(k2)
        print(train_data[k2][:top])

if __name__ == '__main__':
    display_top_train_data(3)