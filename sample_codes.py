import torch
import random
from torch import nn,optim
import math
import numpy as np
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from sklearn.metrics import roc_auc_score
from PIL import Image
import pandas as pd
import random
import sklearn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



# cnn model, the main part of the architecture were refered from https://www.doczamora.com/cats-vs-dogs-binary-classifier-with-pytorch-cnn
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.dropout = nn.Dropout(p=0.2)
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(5,5),stride=2,padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(5,5),stride=2,padding=1)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=1)
        #self.conv4 = nn.Conv2d(in_channels=24,out_channels=48,kernel_size=4)
        #self.conv5 = nn.Conv2d(in_channels=16,out_channels=20,kernel_size=4)
        
        self.fc1 = nn.Linear(in_features=64*6*6,out_features=500)
        self.fc2 = nn.Linear(in_features=500,out_features=100)
        self.fc3 = nn.Linear(in_features=100,out_features=50)
        self.fc4 = nn.Linear(in_features=50,out_features=2)
        #self.fc5 = nn.Linear(in_features=10,out_features=2)
        
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = F.max_pool2d(F.relu(self.conv3(x)),2)
        #x = self.pool(F.relu(self.conv4(x)))
        #x = self.pool(F.relu(self.conv5(x)))

        x = x.view(x.shape[0],-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        #x = self.dropout(F.relu(self.fc4(x)))
        #x = self.fc5(x)
        
        return x

choice_set = [0,1,2,3,4,5,6,7]
#choice_set = [0,1]
valid_acc_set = []
auc_set = []
accuracy_set = []
f1_set = []
train_loss_set = []
valid_loss_set = []
for choice in choice_set:
    # set seed
    seeding = 16
    torch.manual_seed(seeding)
    torch.cuda.manual_seed(seeding)
    torch.cuda.manual_seed_all(seeding)
    np.random.seed(seeding)
    random.seed(seeding)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    print('choice: ',choice)

    train_dir = './training_set/training_set_original/'
    valid_dir = './validation_set/validation_set_original/'
    test_dir = './test_set/test_set_original/'

    transform1 = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5],[0.5])])
    train_set_1 = datasets.ImageFolder(train_dir,transform=transform1)
    valid_set = datasets.ImageFolder(valid_dir,transform=transform1)
    test_set = datasets.ImageFolder(test_dir,transform=transform1)
    
    if choice == 0:
        train_set = train_set_1
    if choice == 1:
        augmentation = transforms.Compose([transforms.Resize((224,224)),
                                           transforms.RandomHorizontalFlip(p=0.9),
                                           transforms.RandomVerticalFlip(p=0.9),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5],[0.5])])
        train_set_2 = datasets.ImageFolder(train_dir,transform=augmentation)
        train_set = torch.utils.data.ConcatDataset([train_set_1,train_set_2])
    elif choice == 2:
        augmentation = transforms.Compose([transforms.Resize((224,224)),
                                           transforms.RandomResizedCrop(224,scale=(0.6,1)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5],[0.5])])
        train_set_2 = datasets.ImageFolder(train_dir,transform=augmentation)
        train_set = torch.utils.data.ConcatDataset([train_set_1,train_set_2])
    elif choice == 3:
        augmentation = transforms.Compose([transforms.Resize((224,224)),
                                           transforms.ColorJitter(brightness=0.5,
                                                                  contrast=0.5,
                                                                  hue=0.2),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5],[0.5])])
        train_set_2 = datasets.ImageFolder(train_dir,transform=augmentation)
        train_set = torch.utils.data.ConcatDataset([train_set_1,train_set_2])
    elif choice == 4:
        augmentation = transforms.Compose([transforms.Resize((224,224)),
                                           transforms.RandomHorizontalFlip(p=0.9),
                                           transforms.RandomVerticalFlip(p=0.9),
                                           transforms.RandomResizedCrop(224,scale=(0.6,1)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5],[0.5])])
        train_set_2 = datasets.ImageFolder(train_dir,transform=augmentation)
        train_set = torch.utils.data.ConcatDataset([train_set_1,train_set_2])
    elif choice == 5:
        augmentation = transforms.Compose([transforms.Resize((224,224)),
                                           transforms.RandomHorizontalFlip(p=0.9),
                                           transforms.RandomVerticalFlip(p=0.9),
                                           transforms.ColorJitter(brightness=0.5,
                                                                  contrast=0.5,
                                                                  hue=0.2),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5],[0.5])])
        train_set_2 = datasets.ImageFolder(train_dir,transform=augmentation)
        train_set = torch.utils.data.ConcatDataset([train_set_1,train_set_2])
    elif choice == 6:
        augmentation = transforms.Compose([transforms.Resize((224,224)),
                                           transforms.RandomResizedCrop(224,scale=(0.6,1)),
                                           transforms.ColorJitter(brightness=0.5,
                                                                  contrast=0.5,
                                                                  hue=0.2),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5],[0.5])])
        train_set_2 = datasets.ImageFolder(train_dir,transform=augmentation)
        train_set = torch.utils.data.ConcatDataset([train_set_1,train_set_2])
    elif choice == 7:
        augmentation = transforms.Compose([transforms.Resize((224,224)),
                                           transforms.RandomHorizontalFlip(p=0.9),
                                           transforms.RandomVerticalFlip(p=0.9),
                                           transforms.RandomResizedCrop(224,scale=(0.6,1)),
                                           transforms.ColorJitter(brightness=0.5,
                                                                  contrast=0.5,
                                                                  hue=0.2),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5],[0.5])])
        train_set_2 = datasets.ImageFolder(train_dir,transform=augmentation)
        train_set = torch.utils.data.ConcatDataset([train_set_1,train_set_2])

    
    batch_size = 100
    train_set = DataLoader(dataset=train_set,shuffle=True,batch_size=batch_size)
    valid_set = DataLoader(dataset=valid_set,batch_size=batch_size)
    test_set = DataLoader(dataset=test_set,batch_size=batch_size)
    print('training samples: ',len(train_set.dataset))
    device = torch.device("cuda" if torch.cuda.is_available() == True else "cpu")
    print(device)
    
    net = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.001,weight_decay=1e-05)
    
    all_epochs = 15

    net.train()
    min_valid_loss = 1
    sub_valid_acc_set = []
    sub_valid_loss_set = []
    sub_train_loss_set = []
    for epoch in range(all_epochs):
        
        # training part
        net.train()
        correct_count = 0
        train_loss = 0
        for i, (inputs,labels) in enumerate(train_set):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            output = net(inputs)
    
            output_label = torch.argmax(output,dim=1)
            correct_count += (labels==output_label).sum().item()
            loss = criterion(output,labels)
            train_loss += loss.item()*inputs.size(0)
            loss.backward()
            optimizer.step()
        train_loss = train_loss/len(train_set.dataset)
        print('Epoch: {}, training loss: {}, training accuracy: {}'.format(epoch,round(train_loss,3),round(correct_count/len(train_set.dataset),3)))
        sub_train_loss_set.append(train_loss)
        # validation part
        net.eval()
        correct_count_valid = 0
        valid_loss = 0
        for inputs_valid,labels_valid in valid_set:
            inputs_valid,labels_valid = inputs_valid.to(device),labels_valid.to(device)
            output_valid = net(inputs_valid)
            loss = criterion(output_valid,labels_valid)
            valid_loss += loss.item()*inputs_valid.size(0)
            output_valid_class = torch.argmax(output_valid,dim=1)
            correct_count_valid += (output_valid_class==labels_valid).sum().item()
        valid_loss = valid_loss/len(valid_set.dataset)
        sub_valid_loss_set.append(valid_loss)
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
        valid_acc = correct_count_valid/len(valid_set.dataset)
        print('Epoch: {}, valid loss: {}, valid accuracy: {}'.format(epoch,round(valid_loss,3),round(valid_acc,3)))
        sub_valid_acc_set.append(valid_acc)
        if valid_loss-min_valid_loss > 0.15: #early stopping
            break

    valid_acc_set.append(sub_valid_acc_set)
    train_loss_set.append(sub_train_loss_set)
    valid_loss_set.append(sub_valid_loss_set)
    with torch.no_grad():
        net.eval()
        correct_count = 0
        true_labels_list = []
        predicted_labels_list = []
        
        for inputs,labels in test_set:
            true_labels_list += list(labels.cpu().detach().numpy())
            labels = labels.to(device)
            outputs = net(inputs.to(device))
            loss = criterion(outputs,labels)
            output_label = torch.argmax(outputs,dim=1)
            correct_count += (labels==output_label).sum().item()
            predicted_labels_list += list(output_label.cpu().detach().numpy())
        
        print('test accuracy: {}'.format(round(correct_count/len(test_set.dataset),3)))
        auc = roc_auc_score(true_labels_list,predicted_labels_list)
        f1 = sklearn.metrics.f1_score(true_labels_list,predicted_labels_list)
        auc_set.append(auc)
        f1_set.append(f1)
        test_acc = correct_count/len(test_set.dataset)
        accuracy_set.append(test_acc)
        print('AUROC: ',auc)
        print('F1 score: ',f1)

#print(auc_set)
#print(accuracy_set)
#print(f1_set)
#df = pd.DataFrame(choice_set)
#df[1] = auc_set
#df[2] = accuracy_set
#df[3] = f1_set
#df.columns = ['choice','auc','accuracy','F1']
#df.to_csv('augmentation_performance.txt',sep='\t',index=False)

for i in range(len(valid_acc_set)):
    if len(valid_acc_set[i]) < all_epochs:
        for j in range(all_epochs-len(valid_acc_set[i])):
            valid_acc_set[i].append('None')

for i in range(len(train_loss_set)):
    if len(train_loss_set[i]) < all_epochs:
        for j in range(all_epochs-len(train_loss_set[i])):
            train_loss_set[i].append('None')

for i in range(len(valid_loss_set)):
    if len(valid_loss_set[i]) < all_epochs:
        for j in range(all_epochs-len(valid_loss_set[i])):
            valid_loss_set[i].append('None')

#df1 = pd.DataFrame(valid_acc_set)
#df1.to_csv('valid_acc_record.txt',sep='\t')

#df_loss1 = pd.DataFrame(train_loss_set)
#df_loss2 = pd.DataFrame(valid_loss_set)
#df_loss1.to_csv('train_loss.txt',sep='\t')
#df_loss2.to_csv('valid_loss.txt',sep='\t')

fig,(ax1,ax2) = plt.subplots(2,1)
valid_min_loss_set = []
for subset in valid_loss_set:
    for i in range(len(subset)):
        if subset[i] == 'None':
            subset[i] = 1
    valid_min_loss_set.append(min(subset))

ax1.plot(list(range(8)),valid_min_loss_set,marker='o')
ax2.plot(list(range(8)),accuracy_set,marker='o')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax1.set_title('Loss in the validation set')
ax2.set_title('Accuracy in the test set')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Accuracy')
ax2.set_xlabel('Augmentation experiments')
ax1.set_xticks(list(range(8)))
ax1.set_xticklabels(['O','F','Cr','Co','F+Cr','F+Co','Cr+Co','F+Cr+Co'])
ax1.set_ylim(0.43,0.5)
ax2.set_xticks(list(range(8)))
ax2.set_xticklabels(['O','F','Cr','Co','F+Cr','F+Co','Cr+Co','F+Cr+Co'])
ax2.set_ylim(0.73,0.85)
ax1.text(-1.3,0.505,'A',fontsize=14)
ax2.text(-1.3,0.86,'B',fontsize=14)

plt.tight_layout(pad=1)
plt.show()

#fig.savefig('performance.png',dpi=300,format='png')




