import torch.optim as optim
import torch.nn as nn
import torch
from dataset import MemexQA_simple
import pandas as pd
from model import LinearModel, SimpleLSTMModel, NewFusionModel
import time
import os
import numpy as np


def main():
    cuda = torch.cuda.is_available()
    num_workers = 8 if cuda else 0
    # random initial embedding matrix for new words
    hyperparams = {'batch_size': 64, 'momentum': 1e-2, 'lr': 1e-2, 'lr_stepsize': 3, 'lr_decay': 0.85, 'weight_decay': 5e-6, 'epochs': 10, 'valid_every': 100}
    train_shared = pd.read_pickle('prepro_v1.1/train_shared.p')
    nonglove_dict = {word: np.random.normal(0, 1, 100) for word in train_shared['wordCounter'] if word not in train_shared['word2vec']}
    train_shared['word2vec'].update(nonglove_dict)
    
    
    val_shared = pd.read_pickle('prepro_v1.1/val_shared.p')
    val_nonglove_dict = {word: np.random.normal(0, 1, 100) for word in val_shared['wordCounter'] if word not in val_shared['word2vec']}
    val_shared['word2vec'].update(val_nonglove_dict)
    
    train_data = MemexQA_simple(data=pd.read_pickle('prepro_v1.1/train_data.p'), shared=train_shared)
    valid_data = MemexQA_simple(data=pd.read_pickle('prepro_v1.1/val_data.p'), shared=val_shared)
    test_data = MemexQA_simple(data=pd.read_pickle('prepro_v1.1/test_data.p'), shared=pd.read_pickle('prepro_v1.1/test_shared.p'))

# random initial embedding matrix for new words
# config.emb_mat = np.array([idx2vec_dict[idx] if idx2vec_dict.has_key(idx) 
# else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size)) 
# for idx in xrange(config.word_vocab_size)],dtype="float32") 


    train_loader_args = dict(shuffle=True, batch_size=hyperparams['batch_size'], num_workers=num_workers, pin_memory=True) if cuda\
        else dict(shuffle=True, batch_size=hyperparams['batch_size']//4)
    train_loader = torch.utils.data.DataLoader(train_data, **train_loader_args)

    valid_loader_args = dict(batch_size=hyperparams['batch_size'], num_workers=num_workers, pin_memory=True) if cuda\
        else dict(shuffle=False, batch_size=hyperparams['batch_size']//4)
    valid_loader = torch.utils.data.DataLoader(valid_data, **valid_loader_args)

    test_loader_args = dict(batch_size=hyperparams['batch_size'], num_workers=num_workers, pin_memory=True) if cuda\
        else dict(shuffle=True, batch_size=hyperparams['batch_size']//4)
    test_loader = torch.utils.data.DataLoader(test_data, **test_loader_args)


    # initialize model
    device = torch.device("cuda" if cuda else "cpu")
    #model = SimpleLSTMModel(100, 64, hyperparams['batch_size'], 2, device)
    # input_size, hidden_size, batch_size, num_layers, device, q_linear_size, img_linear_size, multimodal_out, kernel, stride
    model = NewFusionModel(100, 64, hyperparams['batch_size']//4, 2, device, 64, 64, 4, 3, 1)
    model.to(device)

    # setup optim and loss

    criterion = nn.CrossEntropyLoss()
    #optimizer= optim.SGD(model.parameters(), momentum=hyperparams['momentum'], lr = hyperparams['lr'], weight_decay= hyperparams['weight_decay'])
    optimizer= optim.Adam(model.parameters(), lr = hyperparams['lr'], weight_decay= hyperparams['weight_decay'])

    scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=hyperparams['lr_stepsize'], gamma=hyperparams['lr_decay'])

    # start training
    print("Starting training phase......")
    #b_i = 0
    avg_loss = 0
    for i in range(hyperparams['epochs']):
        start = time.time()
        model.train()
        n_correct,n_total = 0, 0
        batch_count = 0
        for j, (batch_data, batch_labels) in enumerate(train_loader):
            if j == len(train_loader) - 1:
                break
            optimizer.zero_grad()

            #batch_data = batch_data.to(device)
            #batch_labels = batch_labels.to(device)

            output = model(batch_data)
            #loss = criterion(output, batch_labels.long().cuda())
            loss = criterion(output, batch_labels.long().to(device))
            #print("shape out:", output.shape)
            # train accuracy 
            res = torch.argmax(output, 1)
            res = res.to(device)
            #n_correct += (res == batch_labels.cuda()).sum().item()
            n_correct += (res == batch_labels).sum().item()
            n_total += len(batch_data['q_vec'])
            avg_loss += loss
            batch_count += 1
            loss.backward()
            optimizer.step()


            # # validate and save model 
            # b_i += 1
            # # evaluate performance on validation set periodically
            # if b_i % hyperparams['valid_every'] == 0:
            #     model.eval()
            #     with torch.no_grad():
            #         losses = []
            #         valid_correct, valid_loss = 0, 0
            #         # validation for classification
            #         for (b_data, b_label) in valid_loader:
            #             b_data = b_data.to(device)
            #             b_label = b_label.to(device)
            #             feat, outp = model(b_data)
            #             resm = torch.argmax(outp, axis=1)
            #             resm = resm.to(device)
            #             correct = (resm == b_label).sum().item()
            #             valid_correct += correct
            #             loss = criterion(outp, b_label.long())
            #             losses.append(lo)
            #         if len(losses) == 1:
            #             losses = losses[0]
            #         else:
            #             losses = torch.tensor(losses)
            #         valid_accu = valid_correct / len(valid_labels)
            #         valid_loss = torch.mean(losses)
            #         print(f"VALID ===> Epoch {i}, took time {time.time()-start:.1f}s, valid accu: {valid_accu:.4f}, valid loss: {valid_loss:.6f}")
            #     model.train()
        train_acc = n_correct * 100/n_total
        avg_loss = avg_loss / batch_count
        print(f"TRAIN ===> Epoch {i}, took time {time.time()-start:.1f}s, train accu: {train_acc:.4f}, train loss: {loss:.6f}")
        #print(f"predicted: {res},  actual: {batch_labels}")
        
    #     snapshot_prefix = os.path.join(os.getcwd(), 'snapshot/')
    #     if(not os.path.exists(snapshot_prefix)):
    #         os.mkdir(snapshot_prefix)
            
    #     scheduler1.step()
    #     torch.save({
    #                 'model_state_dict': model.state_dict(),
    #                 'optimizer_label_state_dict': optimizer.state_dict(),
    #                 'scheduler1_state_dict' : scheduler1.state_dict()
    #     }, snapshot_prefix + "Model_"+str(i))
    #     #print(f"TRAIN ===> Epoch {i}, took time {time.time()-start:.1f}s, train accu: {train_acc:.4f}, train loss: {loss:.6f}")
    # final_model_prefix = os.path.join(os.getcwd(), 'final/')
    # if(not os.path.exists(final_model_prefix)):
    #     os.mkdir(final_model_prefix)
    # final_path = final_model_prefix + 'final_model.pt'
    # torch.save(model, final_path)


    # predict
    # print("=======")
    # print("Classifcation Testing Phase......")
    # model.eval()
    # with torch.no_grad(), open(classification_output, 'w') as f:
    #     i = 0
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow(["id","label"])
    #     for (batch_data, _) in test_loader:
    #         batch_data = batch_data.to(device)
    #         _, output = model(batch_data)
    #         prediction = torch.argmax(output, axis=1)
    #         for p in prediction.data:
    #             writer.writerow([i, p.item()])
    #             i += 1
    
if __name__ == '__main__':
    main()
