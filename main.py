import torch.optim as optim
import torch.nn as nn
import torch
from dataset import MemexQA
import pandas as pd





def main():
    cuda = torch.cuda.is_available()
    num_workers = 8 if cuda else 0


    train_data = MemexQA(data=pd.read_pickle('prepro_v1.1/train_data.p'), shared=pd.read_pickle('prepro_v1.1/train_shared.p'))
    valid_data = MemexQA(data=pd.read_pickle('prepro_v1.1/val_data.p'), shared=pd.read_pickle('prepro_v1.1/val_shared.p'))
    test_data = MemexQA(data=pd.read_pickle('prepro_v1.1/test_data.p'), shared=pd.read_pickle('prepro_v1.1/test_shared.p'))


    train_loader_args = dict(shuffle=True, batch_size=hyperparams['batch_size'], num_workers=num_workers, pin_memory=True) if cuda\
        else dict(shuffle=True, batch_size=hyperparams['batch_size']//4)
    train_loader = torch.utils.data.DataLoader(train_data, **train_loader_args)

    valid_loader_args = dict(batch_size=hyperparams['batch_size'], num_workers=num_workers, pin_memory=True) if cuda\
        else dict(batch_size=hyperparams['batch_size']//4)
    valid_loader = torch.utils.data.DataLoader(valid_data, **valid_loader_args)

    test_loader_args = dict(batch_size=hyperparams['batch_size'], num_workers=num_workers, pin_memory=True) if cuda\
        else dict(batch_size=hyperparams['batch_size']//4)
    test_loader = torch.utils.data.DataLoader(test_data, **test_loader_args)


    # initialize model
    model = FVTA(...)

    device = torch.device("cuda" if cuda else "cpu")
    model.to(device)


    # setup optim and loss

    criterion = nn.CrossEntropyLoss()
    optimizer= optim.SGD(model.parameters(), momentum=hyperparams['momentum'], lr = hyperparams['lr'], weight_decay= hyperparams['weight_decay'])
    scheduler1 = optim.lr_scheduler.StepLR(optimizer_label, step_size=hyperparams['lr_stepsize'], gamma=hyperparams['lr_decay'])

    # start training
    print("Starting training phase......")
    auc_scores = []
    b_i = 0
    for i in range(hyperparams['epochs']):
        start = time.time()
        model.train()
        loss = 0
        n_correct,n_total = 0, 0
        for (batch_data, batch_labels) in train_loader:
            optimizer.zero_grad()

            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            features, output = model(batch_data)
            l_loss = criterion(output, batch_labels.long())
            loss = l_loss + hyperparams['center_loss_lambda'] * c_loss

            # train accuracy 
            res = torch.argmax(output, 1)
            res = res.to(device)
            n_correct += (res == batch_labels).sum().item()
            n_total += len(batch_data)

            loss.backward()
            optimizer.step()


            # validate and save model 
            b_i += 1
            # evaluate performance on validation set periodically
            if b_i % hyperparams['valid_every'] == 0:
                model.eval()
                with torch.no_grad():
                    losses = []
                    valid_correct, valid_loss = 0, 0
                    # validation for classification
                    for (b_data, b_label) in valid_loader:
                        b_data = b_data.to(device)
                        b_label = b_label.to(device)
                        feat, outp = model(b_data)
                        resm = torch.argmax(outp, axis=1)
                        resm = resm.to(device)
                        correct = (resm == b_label).sum().item()
                        valid_correct += correct
                        loss = criterion(outp, b_label.long())
                        losses.append(lo)
                    if len(losses) == 1:
                        losses = losses[0]
                    else:
                        losses = torch.tensor(losses)
                    valid_accu = valid_correct / len(valid_labels)
                    valid_loss = torch.mean(losses)
                    print(f"VALID ===> Epoch {i}, took time {time.time()-start:.1f}s, valid accu: {valid_accu:.4f}, valid loss: {valid_loss:.6f}")
                model.train()
        train_acc = n_correct/n_total
        snapshot_prefix = os.path.join(os.getcwd(), 'snapshot/')
        scheduler1.step()
        torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_label_state_dict': optimizer.state_dict(),
                    'scheduler1_state_dict' : scheduler1.state_dict(),
                    'scheduler2_state_dict' : scheduler.state_dict(),
        }, snapshot_prefix + "Model_"+str(i))
        print(f"TRAIN ===> Epoch {i}, took time {time.time()-start:.1f}s, train accu: {train_acc:.4f}, train loss: {loss:.6f}")
    final_model_prefix = os.path.join(os.getcwd(), 'final/')
    final_path = final_model_prefix + 'final_model.pt'
    torch.save(model, final_path)

    # write auc scores over the epochs to a file
    with open("final/auc.csv", 'w') as f:
        auc_scores = [str(s) for s in auc_scores]
        for scor in auc_scores:
            f.write(scor)

    # predict
    # print("=======")
    # print("Classifcation Testing Phase......")
    model.eval()
    with torch.no_grad(), open(classification_output, 'w') as f:
        i = 0
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["id","label"])
        for (batch_data, _) in test_loader:
            batch_data = batch_data.to(device)
            _, output = model(batch_data)
            prediction = torch.argmax(output, axis=1)
            for p in prediction.data:
                writer.writerow([i, p.item()])
                i += 1
    