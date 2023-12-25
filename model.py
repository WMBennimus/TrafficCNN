# CSCI 646: Deep Learning
# Assignment 1
# Benjamin Netzer

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random
from PIL import Image

EPOCH_COUNT = 200
INPUT_SIZE = 64
LEARNING_RATE = 0.015
DROPOUT = 0.5
MODE = "train"

class TrafficDataset(Dataset):
    def __init__(self,df=None):
        convert_tensor = transforms.ToTensor()
        self.id = []
        self.data = []
        self.day = []
        self.time = []
        self.label = []
        if df is None:
            return
        count = len(df)
        for i, row in df.iterrows():
            img1 = Image.open(os.path.join("samples",row["File 1"]))
            img2 = Image.open(os.path.join("samples",row["File 2"]))
            t1 = convert_tensor(img1)
            t2 = convert_tensor(img2)
            self.data.append(torch.concat([t1,t2]))
            self.id.append(row["ID"])
            self.day.append(row["Day"])
            self.time.append(row["Time"])
            self.label.append(row["Label"]-1)
            print(str(i)+" / "+str(count))
        print("Done!")

    
    def __getitem__(self, i):
        result = {}
        result["id"] = self.id[i]
        result["data"] = self.data[i]
        result["label"] = self.label[i]
        result["day"] = self.day[i]
        result["time"] = self.time[i]
        return result
    
    def __len__(self):
        return len(self.label)

def createDatasets(ds):
    count = len(ds)
    train_size = count*4//5
    indeces = [i for i in range(count)]
    for i in range(count):
        j = random.randint(0, i)
        t = indeces[i]
        indeces[i] = indeces[j]
        indeces[j] = t
    train_rows = indeces[:train_size]
    test_rows = indeces[train_size:]
    
    print(ds.data[0].size())
    
    train_ds = TrafficDataset()
    test_ds = TrafficDataset()
    for i in train_rows:
        train_ds.id.append(ds.id[i])
        train_ds.data.append(ds.data[i])
        train_ds.day.append(ds.day[i])
        train_ds.time.append(ds.time[i])
        train_ds.label.append(ds.label[i])
    for i in test_rows:
        test_ds.id.append(ds.id[i])
        test_ds.data.append(ds.data[i])
        test_ds.day.append(ds.day[i])
        test_ds.time.append(ds.time[i])
        test_ds.label.append(ds.label[i])
    return train_ds, test_ds

def adjust_learning_rate(lr,opt,ep,dec):
    if(ep>30):
        lr = 0.0053
    if(epoch>=50):
        lr = 0.0015
    if(epoch>=100):
        lr = 0.0008
    for param_group in opt.param_groups:
        param_group['lr'] = lr
    return lr

def load_data(DATA_PATH, batch_size):
    ds = None
    if os.path.isfile(DATA_PATH):
        ds = torch.load(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH+".csv")
        ds = TrafficDataset(df)
        torch.save(ds, DATA_PATH)
    
    
    train_ds, test_ds = createDatasets(ds)
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=8)
    
    return train_loader, test_loader

def save_checkpoint(cpath, model, epoch, optimizer, global_step):
    checkpoint = {
        'epoch':epoch,
        'global_step':global_step,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict()}
    torch.save(checkpoint, cpath)

def load_checkpoint(cpath):
    checkpoint = torch.load(cpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoches = checkpoint['epoch']
    
class NNModel(nn.Module):
    def __init__(self):
        super(NNModel,self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(6,32,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),

            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64,128,kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(128,256,kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(56320,5000),
            nn.ReLU()
        )
        self.fclayer = nn.Sequential(
            nn.Linear(5000,128),
            nn.ReLU(),
            nn.Linear(128,5),
            nn.Softmax(dim=1)
        )

    def forward(self, x, day, time):
        x = self.convlayer(x)
        torch.concat([x,day.unsqueeze(1),time.unsqueeze(1)],dim=1)
        x = self.fclayer(x)
        return x


using_cuda = torch.cuda.is_available()
device = torch.device("cuda" if using_cuda else "cpu")
print("Using device "+torch.cuda.get_device_name())
if using_cuda:
    torch.cuda.manual_seed(72)

model = NNModel()
model.to(device)
print("Model loaded")

if __name__ == "__main__":
    
    optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum=0.9)
    loss_fun = nn.CrossEntropyLoss()
    
    train_loader,test_loader = load_data("data_tiny",INPUT_SIZE)
    
    print("Data loaded!")
    
    global_step = 0
    output = []
    if MODE == 'train':
        for epoch in range(EPOCH_COUNT):
            print("Epoch #"+str(epoch+1)+":")
            #LEARNING_RATE = adjust_learning_rate(LEARNING_RATE, optimizer, epoch, 0.1)
            avg_loss = 0
            model = model.train()
            for batch in train_loader:
                global_step += 1
                labels = batch.pop("label").to(device)
                imgs = batch.pop("data").to(device)
                day = batch.pop("day").to(device)
                time = batch.pop("time").to(device)
                optimizer.zero_grad()
                output_y = model(imgs,day,time)
                
                loss = loss_fun(output_y, labels)
                loss.backward()
                optimizer.step()
                avg_loss = avg_loss + loss.item()
            acc = 0
            count = 0
            model = model.eval()
            for batch in test_loader:
                labels = batch.pop("label").to(device)
                imgs = batch.pop("data").to(device)
                day = batch.pop("day").to(device)
                time = batch.pop("time").to(device)
                
                output_y = model(imgs,day,time)
                
                acc += (torch.argmax(output_y,1) == labels).float().sum()
                count += len(labels)
            acc /= count
            print("Accuracy = "+str(acc.item()*100))
            print("Loss = "+str(avg_loss))
            output.append([avg_loss, acc.item()])
    df = pd.DataFrame(data=output, columns=["Loss","Accuracy"])
    df.to_csv("results.csv")
    torch.save(model.state_dict(),"model")
    
    global_step = 0
    output = []
    model = model.eval()
    for batch in train_loader:
        global_step += 1
        elID = batch.pop("id")
        labels = batch.pop("label").to(device)
        imgs = batch.pop("data").to(device)
        day = batch.pop("day").to(device)
        time = batch.pop("time").to(device)
        optimizer.zero_grad()
        output_y = model(imgs,day,time)
        for i in range(labels.size()[0]):
            newRow = [elID[i].item(), labels[i].item(), torch.argmax(output_y[i]).item()+1]
            for j in range(5):
                newRow.append(output_y[i][j].item())
            output.append(newRow)
        
    for batch in test_loader:
        global_step += 1
        elID = batch.pop("id")
        labels = batch.pop("label").to(device)
        imgs = batch.pop("data").to(device)
        day = batch.pop("day").to(device)
        time = batch.pop("time").to(device)
        optimizer.zero_grad()
        output_y = model(imgs,day,time)
        for i in range(labels.size()[0]):
            newRow = [elID[i].item(), labels[i].item(), torch.argmax(output_y[i]).item()+1]
            for j in range(5):
                newRow.append(output_y[i][j].item())
            output.append(newRow)
                
            
    df = pd.DataFrame(data=output, columns=["ID","Label","Prediction","p1","p2","p3","p4","p5"])
    df.to_csv("eval_results.csv")