import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import random_split
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.backends.cudnn as cudnn

from utility import *
from model import get_model
from cutmix import CutMixCriterion
from cutmix import CutMixCollator

if __name__ == "__main__":
    
    ### arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vgg", type=str, help='model name(vgg or resnet)')
    parser.add_argument("--valid", default=1, type=int, help='apply validation set from training set or not')
    parser.add_argument("--less_data", default=1, type=int, help='apply imbalanced training set or not' )
    parser.add_argument("--batch_size", default=128, type=int, help='batch size')
    parser.add_argument("--weight", default=1, type=int, help='apply class weight or not')
    parser.add_argument("--valid_rate", default=0.2, type=float, help='ratio of validation set compared to original training set')
    parser.add_argument("--ROS", default=1, type=int, help='apply random over sampling or not')
    parser.add_argument("--epochs", default=200, type=int, help='epochs for training')
    parser.add_argument("--c", default=1, type=int, help='apply transform RandomCrop or not')
    parser.add_argument("--f", default=1, type=int, help='apply transform RandomFlip or not')
    parser.add_argument("--e", default=1, type=int, help='apply transform RandomErasing or not')
    parser.add_argument("--cutmix", default=0, type=int, help='apply cutmix or not')
    args = parser.parse_args()
    
    print(f'model:{args.model} weight:{args.weight} valid:{args.valid} lessData:{args.less_data} epochs:{args.epochs} batch_size:{args.batch_size} ROS:{args.ROS} crop:{args.c} flip:{args.f} erase:{args.e} cutmix:{args.cutmix}')
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
          "horse", "ship", "truck"]
    
    ### get data, dataset.train = trainset #########  dataset.test = testset
    dataset = Cifar(args.batch_size, 2, args.less_data, args.c, args.f, args.e)
    
    ### get test_loader
    test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    ### If use cutmix, setting the collate_fn for dataloader
    if args.cutmix:
        collator = CutMixCollator(1.0)
    else:
        collator = torch.utils.data.dataloader.default_collate
    
    if args.valid:  ### construct a validation set from train data or not
        train_data, val_data = random_split(dataset.train, [len(dataset.train) - int(args.valid_rate * len(dataset.train)), int(args.valid_rate * len(dataset.train))])
        if args.ROS:   ### apply random over sampling or not
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                                     collate_fn=collator, num_workers=2, sampler=weight_sampler(val_data))
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, 
                                                       collate_fn=collator,num_workers=2, sampler = weight_sampler(train_data))
        else:
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False,collate_fn=collator, num_workers=2)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False,collate_fn=collator, num_workers=2)
    else:
        train_loader = torch.utils.data.DataLoader(dataset.train, batch_size=args.batch_size,collate_fn=collator, shuffle=True, num_workers=2)
        val_loader = test_loader
        
    ### training with class weight or not
    if args.weight:
        weights = torch.tensor([1/5000, 1/5000, 1/2500, 1/5000, 1/2500, 1/5000, 1/5000,1/5000,1/5000, 1/2500], dtype=torch.float32)
        weights = weights / weights.sum()
        weights = weights.cuda()
        if args.cutmix:   ### if cutmix, it needs different criterion
            criterion = CutMixCriterion(reduction='mean', weight=weights, Is_weight = args.weight)
        else:
            criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        if args.cutmix:
            criterion = CutMixCriterion(reduction='mean', weight=0, Is_weight = args.weight)
        else:
            criterion = nn.CrossEntropyLoss()
    
    #### get the model and initilize optimizer and scheduler
    model = get_model(args.model)
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    valid_loss_min = np.Inf
    train_loss_min = np.Inf
    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []
    
    for epoch in range(args.epochs):
        
        ### start training 
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_correct_1 = 0
        train_correct_2 = 0
        train_acc = None
        train_total = 0
        t = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.cuda()
            
            ### additional process for cutmix
            if isinstance(labels, (tuple, list)):
                labels1, labels2, lam = labels
                labels = (labels1.cuda(), labels2.cuda(), lam)
            else:
                labels = labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_total += inputs.size(0)
            _, predicted = outputs.max(1)
            
            ### additional process for cutmix
            if isinstance(labels, (tuple, list)):
                labels1, labels2, lam = labels
                train_correct_1 += predicted.eq(labels1).sum().item()
                train_correct_2 += predicted.eq(labels2).sum().item()
                train_acc = (lam * train_correct_1 + (1 - lam) * train_correct_2) / train_total
                t = int(lam * train_correct_1 + (1 - lam) * train_correct_2)
            else:
                train_correct += predicted.eq(labels).sum().item()
                train_acc = train_correct / train_total
                t = train_correct
        
        ### calculate loss and append 
        train_loss = train_loss / (i+1)
        train_loss_history.append(train_loss)
        train_acc_history.append(100.*train_acc)
        
        ### safe the checkpoint for lowest training loss
        if train_loss < train_loss_min:
            state = {
                'net': model.state_dict(),
                'model': args.model,
                'acc': f'{100.*train_acc}%',
                'weight': args.weight,
                'valid': args.valid,
                'less_data': args.less_data,
                'epoch': args.epochs,
                'ROS': args.ROS,
                'batch_size': args.batch_size,
                'crop': args.c,
                'flip': args.f,
                'erase': args.e,
                'cutmix': args.cutmix,
                }
            torch.save(state, f'./models/train_model:{args.model}_weight:{args.weight}_valid:{args.valid}_lessData:{args.less_data}_epochs:{args.epochs}_batch_size:{args.batch_size}_ROS:{args.ROS}_crop:{args.c}_flip:{args.f}_erase:{args.e}_cutmix:{args.cutmix}.pt')
            #torch.save(state, f'./models/train_lessData:{args.less_data}_crop:{args.c}_flip:{args.f}_erase:{args.e}.pt')
            train_loss_min = train_loss
        
        ### validation part
        model.eval()
        val_loss = 0.
        val_correct = 0
        val_correct_1 = 0
        val_correct_2 = 0
        val_acc = None
        val_total = 0
        v = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.cuda()
                
                ### additional process for cutmix
                if isinstance(labels, (tuple, list)):
                    labels1, labels2, lam = labels
                    labels = (labels1.cuda(), labels2.cuda(), lam)
                else:
                    labels = labels.cuda()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                val_total += inputs.size(0)
                val_loss += loss.item()
                
                ### additional process for cutmix
                if isinstance(labels, (tuple, list)):
                    labels1, labels2, lam = labels
                    val_correct_1 += predicted.eq(labels1).sum().item()
                    val_correct_2 += predicted.eq(labels2).sum().item()
                    val_acc = (lam * val_correct_1 + (1 - lam) * val_correct_2) / val_total
                    v = int(lam * val_correct_1 + (1 - lam) * val_correct_2)
                else:
                    val_correct += predicted.eq(labels).sum().item()
                    val_acc = val_correct / val_total
                    v = val_corrct
        
        ### calculating validation loss and collect loss and accuracy history
        val_loss = val_loss / (i+1)
        val_loss_history.append(val_loss)
        val_acc_history.append(100.*val_acc)
        print('[ epoch:%d ] train loss: %.5f val loss: %.5f train_acc:%.2f%% (%d/%d) val_acc:%.2f%% (%d/%d)' % (epoch + 1, train_loss, val_loss,100.*train_acc,t,train_total, 100.*val_acc, v, val_total))
        
        ### saving checkpoints for lowest validation loss
        if val_loss < valid_loss_min:
            state = {
                'net': model.state_dict(),
                'model': args.model,
                'acc': f'{100.*val_acc}%',
                'weight': args.weight,
                'valid': args.valid,
                'less_data': args.less_data,
                'epoch': args.epochs,
                'ROS': args.ROS,
                'batch_size': args.batch_size,
                'crop': args.c,
                'flip': args.f,
                'erase': args.e,
                'cutmix': args.cutmix,
                }
            torch.save(state, f'./models/val_model:{args.model}_weight:{args.weight}_valid:{args.valid}_lessData:{args.less_data}_epochs:{args.epochs}_batch_size:{args.batch_size}_ROS:{args.ROS}_crop:{args.c}_flip:{args.f}_erase:{args.e}_cutmix:{args.cutmix}.pt')
            
            valid_loss_min = val_loss
        scheduler.step()
        
    ### load the model for testing set evaluation
    descript = torch.load(f'./models/val_model:{args.model}_weight:{args.weight}_valid:{args.valid}_lessData:{args.less_data}_epochs:{args.epochs}_batch_size:{args.batch_size}_ROS:{args.ROS}_crop:{args.c}_flip:{args.f}_erase:{args.e}_cutmix:{args.cutmix}.pt')
    
    ### open a file to record the evaluation values
    f = open(f'./txt/val_acc:{100.*val_acc}%_model:{args.model}_weight:{args.weight}_valid:{args.valid}_lessData:{args.less_data}_epochs:{args.epochs}_batch_size:{args.batch_size}_ROS:{args.ROS}_crop:{args.c}_flip:{args.f}_erase:{args.e}_cutmix:{args.cutmix}.txt', 'w')
    
    ### record the model state for all parameters (class_weight, valid.. .... . .. .)
    nl = '\n'
    for a, b in enumerate(descript):
        if a == 0:
            continue
        f.write(f'{b}:{descript[b]}{nl}')
    
    ### load the model
    model.load_state_dict(descript['net'])
    class_correct = list(0 for i in range(10))
    class_total = list(0 for i in range(10))
    n = 0
    scm = np.zeros((10,10))

    model.eval()
    
    ### test set evaluation
    with torch.no_grad():
        for j, data in enumerate(test_loader):
            images, labels = data[0].cuda(), data[1]
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu()
            cm = confusion_matrix(labels, predicted, labels=list(np.arange(10)))
            scm += cm
            c = (predicted == labels).squeeze()
            n += len(c[c==True])
            for i in range(len(data[1])):
                label = labels[i].item()
                if c[i].item() == True:
                    class_correct[label] += 1
                class_total[label] += 1

    ### record the accuracy for each class and the average accuracy of all
    for i in range(10):
        f.write('Accuracy of %5s : %2f %%\n' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    
    f.write('Average Accuracy: %2f %%\n' % (n/100))
    f.close()
    
    ### save the heatmap
    plt.figure(figsize=(12,12))
    sns.heatmap(scm, annot=True, xticklabels = classes, yticklabels = classes, fmt=".1f", cmap="Blues")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'./images/heatmap_test_acc:{n/100}%_model:{args.model}_weight:{args.weight}_valid:{args.valid}_lessData:{args.less_data}_epochs:{args.epochs}_batch_size:{args.batch_size}_ROS:{args.ROS}_crop:{args.c}_flip:{args.f}_erase:{args.e}_cutmix:{args.cutmix}.png')
    
    
    ### save the loss history
    plt.figure()
    plt.title("Loss History")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(train_loss_history,label="Train loss")
    plt.plot(val_loss_history,label="Val loss")
    plt.legend()
    plt.savefig(f'./images/loss_test_acc:{n/100}%_model:{args.model}_weight:{args.weight}_valid:{args.valid}_lessData:{args.less_data}_epochs:{args.epochs}_batch_size:{args.batch_size}_ROS:{args.ROS}_crop:{args.c}_flip:{args.f}_erase:{args.e}_cutmix:{args.cutmix}.png')
    
    ### save the accuracy history
    plt.figure()
    plt.title("Accuracy History")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(train_acc_history,label="Train acc")
    plt.plot(val_acc_history,label="Val acc")
    plt.legend()
    plt.savefig(f'./images/acc_test_acc:{n/100}%_model:{args.model}_weight:{args.weight}_valid:{args.valid}_lessData:{args.less_data}_epochs:{args.epochs}_batch_size:{args.batch_size}_ROS:{args.ROS}_crop:{args.c}_flip:{args.f}_erase:{args.e}_cutmix:{args.cutmix}.png')
    