from model import NNModel, TrafficDataset, device, model

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random
from PIL import Image
import os

import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from PIL import ImageTk

SAMPLE_DIR = "samples"
DAYS_OF_WEEK = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

mdown = False
mx = 0
my = 0

class EvalWindow(tk.Tk):
    def __init__(self, src_path,dl):
        super().__init__()
        self.dl = dl
        self.title("Project Evaluator O'Clock")
        self.data = []
        self.b1 = tk.Button(self, text="Done", command=self.finish)
        self.b2 = tk.Button(self, text="Cancel", command=self.close)
        self.src_path = src_path
        self.currentImage = 0
        self.samples = pd.read_csv("data.csv")
        self.count = len(self.samples)
        self.lab = tk.Label(self, text="Image ID goes here")
        self.lab.pack()
        self.img1 = tk.Label(self)
        self.img2 = tk.Label(self)
        self.img1.pack()
        self.img2.pack()
        self.b1.pack()
        self.b2.pack()
        self.bind("<Key>", self.start)
        
    def load_image(self):
        if self.currentImage >= self.count:
            self.finish()
            return
        i = 0
        for batch in self.dl:
            self.batch = batch
            if i >= self.currentImage:
                break
            i += 1
        row = None
        for i, r in self.samples.iterrows():
            if batch["id"] == r["ID"]:
                row = r
                break
        self.fnameA = row["File 1"]
        self.fnameB = row["File 2"]
        self.label = row["Label"]
        labels = batch.pop("label").to(device)
        imgs = batch.pop("data").to(device)
        day = batch.pop("day").to(device)
        time = batch.pop("time").to(device)
        out = model(imgs, day, time)
        print(out)
        self.out = torch.argmax(out[0]).item()+1
        self.distribution = [0]*5
        for i, x in enumerate(out[0]):
            self.distribution[i] = x.item()
        i1 = ImageTk.PhotoImage(Image.open(os.path.join(self.src_path,self.fnameA)))
        i2 = ImageTk.PhotoImage(Image.open(os.path.join(self.src_path,self.fnameB)))
        self.img1.config(image=i1)
        self.img1.image = i1
        self.img2.config(image=i2)
        self.img2.image = i2
        self.photoID = self.fnameA.split("_")[0][1:-1]
        self.lab.config(text=("Sample #"+self.photoID+" (Label = "+str(self.label)+", Prediction = "+str(self.out)+")"))
        
        self.currentImage += 1
        self.label_image()
    
    def label_image(self):
        newRow = [self.photoID, self.label, self.out, self.distribution[0], self.distribution[1], self.distribution[2], self.distribution[3], self.distribution[4]]
        print(self.distribution)
        self.data.append(newRow)
        self.load_image()
    
    def start(self, event):
        self.load_image()
    
    def finish(self):
        df = pd.DataFrame(data=self.data,columns=["ID","Result","Label","Traffic:1","Traffic:2","Traffic:3","Traffic:4","Traffic:5"])
        df.to_csv("eval.csv")
        self.close()
    
    def close(self):
        self.destroy()
        
            

if __name__ == "__main__":
    if os.path.isfile("./data"):
        ds = torch.load("./data")
        data_loader = DataLoader(dataset=ds, batch_size=1, shuffle=True, num_workers=8)
    else:
        raise FileNotFoundError("The dataset appears to have vanished.")
    print("Data loaded!")
    win = EvalWindow(SAMPLE_DIR, data_loader)
    win.mainloop()
    '''
    files = os.listdir("./copies")
    sort(files)
    for i in range(len(files)):
        fname = files[i]
        photoID = fname.split("_")[0][1:5]
        newID = int(photoID) - 309
        date = fname.split("_")[1]
        if i%2 == 0:
            new_fname = "S"+str(newID)+fname.split("_")[0][5] + "_" + date
            os.system("mv ./copies/"+fname+" ./copies/"+new_fname)
    '''