import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import random
import os

import tkinter as tk
from PIL import Image
from PIL import ImageTk

SAMPLE_DIR = "my_half"
DAYS_OF_WEEK = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

mdown = False
mx = 0
my = 0

def sort(arr):
    if len(arr) <= 1:
        return
    l = len(arr)//2
    A = arr[:l]
    B = arr[l:]
    sort(A)
    sort(B)
    x = 0
    y = 0
    for i in range(len(arr)):
        if y >= len(B) or x < l and A[x] <= B[y]:
            arr[i] = A[x]
            x += 1
        elif x >= len(A) or A[x] > B[y]:
            arr[i] = B[y]
            y += 1

class EvalWindow(tk.Tk):
    def __init__(self, src_path):
        super().__init__()
        
        self.title("Project Labeler O'Clock")
        self.data = []
        self.b1 = tk.Button(self, text="Done", command=self.finish)
        self.b2 = tk.Button(self, text="Cancel", command=self.close)
        self.src_path = src_path
        self.images = os.listdir(src_path)
        sort(self.images)
        self.count = len(self.images)//2
        for i in range(0,self.count*2,2):
            j = random.randint(0,i/2)*2
            t1 = self.images[i]
            t2 = self.images[i+1]
            self.images[i] = self.images[j]
            self.images[i+1] = self.images[j+1]
            self.images[j] = t1
            self.images[j+1] = t2
        self.lab = tk.Label(self, text="Image ID goes here")
        self.lab.pack()
        self.img1 = tk.Label(self)
        self.img2 = tk.Label(self)
        self.img1.pack()
        self.img2.pack()
        self.b1.pack()
        self.b2.pack()
        self.currentImage = 0
        print(self.images)
        self.load_image()
        self.bind("<Key>", self.label_image)
        
    def load_image(self):
        if self.currentImage >= self.count:
            self.finish()
            return
        self.fnameA = self.images[self.currentImage*2]
        self.fnameB = self.images[self.currentImage*2+1]
        i1 = ImageTk.PhotoImage(Image.open(os.path.join(self.src_path,self.fnameA)))
        i2 = ImageTk.PhotoImage(Image.open(os.path.join(self.src_path,self.fnameB)))
        self.img1.config(image=i1)
        self.img1.image = i1
        self.img2.config(image=i2)
        self.img2.image = i2
        self.photoID = self.fnameA.split("_")[0][1:-1]
        fullDate = self.fnameA.split("_")[1][0:-4].split("-")
        self.day = (int(fullDate[1]) - 5)%7
        self.hour = int(fullDate[2])
        self.minute = int(fullDate[3])
        self.lab.config(text=("Sample #"+self.photoID+" ("+DAYS_OF_WEEK[self.day]+" "+str(self.hour)+":"+("0" if self.minute<10 else "")+str(self.minute)+")"))
        self.currentImage += 1
    
    def label_image(self, event):
        val = 0
        try:
            val = int(event.char)
        except:
            pass
        if val >= 0 and val <= 5:
            if val == 0:
                os.system("rm "+os.path.join(self.src_path,self.fnameA))
                os.system("rm "+os.path.join(self.src_path,self.fnameB))
            else:    
                newRow = [self.photoID, self.fnameA, self.fnameB, self.day, (self.hour*60) + self.minute, val]
                print(newRow)
                self.data.append(newRow)
            self.load_image()
    
    def finish(self):
        df = pd.DataFrame(data=self.data,columns=["ID","File 1","File 2","Day","Time","Label"])
        df.to_csv("data.csv")
        self.close()
    
    def close(self):
        self.destroy()
        
            

if __name__ == "__main__":
    win = EvalWindow(SAMPLE_DIR)
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