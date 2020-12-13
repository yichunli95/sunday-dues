import torch.optim as optim
import torch.nn as nn
import torch
from new_dataset_checked_by_hongyuan import MemexQA_new, train_collate
import pandas as pd
from model import LinearModel, SimpleLSTMModel, NewFusionModel
import time
import os
import numpy as np
import argparse
import csv

# hyperparams
EPOCHS = 10
BATCH_SIZE = 64

# optimizer-related
MOMENTUM = 1e-2
LR = 1e-2
LR_STEPSIZE = 3
LR_DECAY = 0.85
WD = 5e-6



def main(train_data_pth, train_shared_pth, val_data_pth, val_shared_pth, test_data_pth, test_shared_pth, isTrain):
    cuda = torch.cuda.is_available()
    num_workers = 8 if cuda else 0
    print("Loading data......")
    start = time.time()
    # load data
    # train_shared = pd.read_pickle('prepro_v1.1/train_shared.p')
    train_shared = pd.read_pickle(train_shared_pth)
    # random initial embedding matrix for new words
    nonglove_dict = {word: np.random.normal(0, 1, 100) for word in train_shared['wordCounter'] if word not in train_shared['word2vec']}
    train_shared['word2vec'].update(nonglove_dict)
    
    val_shared = pd.read_pickle(val_shared_pth)
    val_nonglove_dict = {word: np.random.normal(0, 1, 100) for word in val_shared['wordCounter'] if word not in val_shared['word2vec']}
    val_shared['word2vec'].update(val_nonglove_dict)

    test_shared = pd.read_pickle(test_shared_pth)
    test_nonglove_dict = {word: np.random.normal(0, 1, 100) for word in test_shared['wordCounter'] if word not in test_shared['word2vec']}
    test_shared['word2vec'].update(test_nonglove_dict)

    # train_data = MemexQA_new(data=pd.read_pickle('prepro_v1.1/train_data.p'), shared=train_shared, info=None)
    # valid_data = MemexQA_new(data=pd.read_pickle('prepro_v1.1/val_data.p'), shared=val_shared, info=None)
    # test_data = MemexQA_new(data=pd.read_pickle('prepro_v1.1/test_data.p'), shared=test_shared, info=None)
    train_data = MemexQA_new(data=pd.read_pickle(train_data_pth), shared=train_shared)
    valid_data = MemexQA_new(data=pd.read_pickle(val_data_pth), shared=val_shared)
    test_data = MemexQA_new(data=pd.read_pickle(test_data_pth), shared=test_shared)

    # random initial embedding matrix for new words
    # config.emb_mat = np.array([idx2vec_dict[idx] if idx2vec_dict.has_key(idx) 
    # else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size)) 
    # for idx in xrange(config.word_vocab_size)],dtype="float32") 

    train_loader_args = dict(shuffle=True, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True, collate_fn=train_collate) if cuda\
        else dict(shuffle=True, batch_size=BATCH_SIZE, collate_fn=train_collate)
    train_loader = torch.utils.data.DataLoader(train_data, **train_loader_args)

    valid_loader_args = dict(batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True, collate_fn=train_collate) if cuda\
        else dict(shuffle=False, batch_size=BATCH_SIZE, collate_fn=train_collate)
    valid_loader = torch.utils.data.DataLoader(valid_data, **valid_loader_args)

    test_loader_args = dict(batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True, collate_fn=train_collate) if cuda\
        else dict(shuffle=False, batch_size=BATCH_SIZE, collate_fn=train_collate)
    test_loader = torch.utils.data.DataLoader(test_data, **test_loader_args)
    print(f"Loading data took {time.time() - start:.1f} seconds")
    
    # initialize model
    device = torch.device("cuda" if cuda else "cpu")
    #model = SimpleLSTMModel(100, 64, hyperparams['batch_size'], 2, device)
    # input_size, hidden_size, batch_size, num_layers, device, q_linear_size, img_linear_size, multimodal_out, kernel, stride
    model = NewFusionModel(100, 64, BATCH_SIZE, 2, device, 64, 64, 4, 3, 1)
    model.to(device)

    # setup optim and loss

    criterion = nn.CrossEntropyLoss()
    #optimizer= optim.SGD(model.parameters(), momentum=hyperparams['momentum'], lr = hyperparams['lr'], weight_decay= hyperparams['weight_decay'])
    optimizer= optim.Adam(model.parameters(), lr = LR, weight_decay= WD)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEPSIZE, gamma=LR_DECAY)

    # training
    print("Starting training......")
    for i in range(EPOCHS):
        start = time.time()
        model.train()
        n_correct,n_total = 0, 0
        batch_count = 0
        t_loss = 0
        for j, (batch_data, batch_labels) in enumerate(train_loader):
            if j == len(train_loader) - 1:
                break
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_labels.long().to(device))
            t_loss += loss.item()
            res = torch.argmax(output, 1)
            res = res.to(device)
            n_correct += (res == batch_labels).sum().item()
            n_total += len(batch_data['q_vec'])
            batch_count += 1
            loss.backward()
            optimizer.step()
        train_acc = n_correct / n_total
        train_loss = t_loss / batch_count
        print(f"TRAIN ===> Epoch {i}, took time {time.time()-start:.1f}s, train accu: {train_acc:.4f}, train loss: {train_loss:.6f}")
        scheduler.step()
        
        # validate and save model 
        print("Start validation......")
        start = time.time()
        with torch.no_grad():
            model.eval()            
            valid_correct, loss, num_of_batches, num_of_val = 0, 0, 0, 0
            # validation for classification
            for (vb_data, vb_label) in valid_loader:
                v_output = model(vb_data)
                resm = torch.argmax(v_output, axis=1)
                resm = resm.to(device)
                correct = (resm == vb_label).sum().item()
                valid_correct += correct
                loss += criterion(v_output, vb_label.long().to(device)).item()
                num_of_batches += 1
                num_of_val += vb_label.shape[0]
            val_loss = loss / num_of_batches
            val_accu = valid_correct / num_of_val
        print(f"VALID ===> Epoch {i}, took time {time.time()-start:.1f}s, valid accu: {val_accu:.4f}, valid loss: {val_loss:.6f}")
        
        snapshot_prefix = os.path.join(os.getcwd(), 'snapshot/')
        if not os.path.exists(snapshot_prefix):
            os.makedirs(snapshot_prefix)
        torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict' : scheduler.state_dict(),
        }, snapshot_prefix + "Model_"+str(i))
    
    # testing
    if not isTrain:
        print("Start testing......")
        start = time.time()
        model.eval()
        with torch.no_grad(), open('test_predictions.csv', 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(["predict","actual"])
            for (tbatch_data, tbatch_data_labels) in test_loader:
                test_out = model(tbatch_data)
                predict = torch.argmax(test_out, axis=1)
                correct = (predict == tbatch_data_labels).sum().item()
                for (pred, actual) in zip(predict, correct):
                    writer.writerow([pred, actual])
        print(f"Testing took {time.time()-start:.1f}s")
    print("Finished")
                
                                                    
    
if __name__ == '__main__':
    main('prepro_v1.1/train_data.p',
        'prepro_v1.1/train_shared.p',
        'prepro_v1.1/val_data.p',
        'prepro_v1.1/val_shared.p',
        'prepro_v1.1/test_data.p',
        'prepro_v1.1/test_shared.p',
        isTrain = True)
    # parser = argparse.ArgumentParser(description='Get the train-val-test dataset files')
    
    # parser.add_argument("-td" , "--train_data_pth", help="Enter train data path", type=str)
    # parser.add_argument("-tds", "--train_shared_pth", help="Enter train_shared data path", type=str)
    # parser.add_argument("-vd", "--val_data_pth", help="Enter val data path", type=str)
    # parser.add_argument("-vds", "--val_shared_pth", help="Enter val_shared data path", type=str)
    # parser.add_argument("-test", "--test_data_pth", help="Enter test data path", type=str)
    # parser.add_argument("-test_shared", "--test_shared_pth", help="Enter test_shared data path", type=str)
    # # parser.add_argument("-album", "--album_data_pth", help="Enter album_json data path", type=str)
    # parser.add_argument("isTrain", help="Set True if model is training", type=bool)
    
    # args = parser.parse_args()
    
    # main(args.train_data_pth,
    #     args.train_shared_pth,
    #     args.val_data_pth,
    #     args.val_shared_pth,
    #     args.test_data_pth,
    #     args.test_shared_pth,
    #     isTrain = args.isTrain)


