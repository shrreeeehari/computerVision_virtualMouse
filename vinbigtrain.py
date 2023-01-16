#dependencies
import numpy as np, pandas as pd, torch, IProgress
from glob import glob
import shutil, os
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import yaml
from sklearn.model_selection import GroupKFold
from tqdm.notebook import tqdm
from sklearn import datasets, model_selection
from sklearn.model_selection import train_test_split
from metaflow import FlowSpec,Parameter, Flow, step, get_metadata, kubernetes, S3, IncludeFile, retry
import os.path
from os.path import exists


class VinBig(FlowSpec):
    #data = IncludeFile("data", default="./train.csv")
    
    @step
    def start(self):
        self.dim = 512
        self.next(self.dataload)

    @step
    def dataload(self):
        self.train_df = pd.read_csv("train.csv",nrows=500)

        self.train_df['image_path'] = f'../512data/train/'+(self.train_df).image_id+('.png')

        self.train_df = self.train_df[self.train_df.class_id!=14].reset_index(drop = True)

        print("Training on",(self.train_df).image_id.nunique(),"images")
        self.next(self.preprocess)

    @step
    def preprocess(self):
        #Pre-Processing
        self.train_df['x_min'] = self.train_df.apply(lambda row: (row.x_min)/row.width, axis =1)
        self.train_df['y_min'] = self.train_df.apply(lambda row: (row.y_min)/row.height, axis =1)

        self.train_df['x_max'] = self.train_df.apply(lambda row: (row.x_max)/row.width, axis =1)
        self.train_df['y_max'] = self.train_df.apply(lambda row: (row.y_max)/row.height, axis =1)

        self.train_df['x_mid'] = self.train_df.apply(lambda row: (row.x_max+row.x_min)/2, axis =1)
        self.train_df['y_mid'] = self.train_df.apply(lambda row: (row.y_max+row.y_min)/2, axis =1)

        self.train_df['w'] = self.train_df.apply(lambda row: (row.x_max-row.x_min), axis =1)
        self.train_df['h'] = self.train_df.apply(lambda row: (row.y_max-row.y_min), axis =1)

        self.train_df['area'] = self.train_df['w']*self.train_df['h']
        #print(self.train_df.head())

        features = ['x_min', 'y_min', 'x_max', 'y_max', 'x_mid', 'y_mid', 'w', 'h', 'area']
        self.X = self.train_df[features]
        self.y = self.train_df['class_id']
        #(self.X).shape, (self.y).shape

        self.class_ids, self.class_names = list(zip(*set(zip(self.train_df.class_id, self.train_df.class_name))))
        self.classes = list(np.array(self.class_names)[np.argsort(self.class_ids)])
        self.classes = list(map(lambda x: str(x), self.classes))
        #print(self.classes)

        self.next(self.datasplit)

    @step
    def datasplit(self):
        
        self.train_data, self.val_data = model_selection.train_test_split(self.train_df, test_size = 0.30)
        self.train_df.head()

        self.train_files = []
        self.val_files   = []
        self.val_files += list(self.val_data.image_path.unique())
        self.train_files += list(self.train_data.image_path.unique())

        os.makedirs('labels/train', exist_ok = True)
        os.makedirs('labels/val', exist_ok = True)
        os.makedirs('images/train', exist_ok = True)
        os.makedirs('images/val', exist_ok = True)
        label_dir = '../labels'

        for file in (self.train_files):
            shutil.copy(file, 'images/train')
            filename = file.split('/')[-1].split('.')[0]
            shutil.copy(os.path.join(label_dir, filename+'.txt'), 'labels/train')
            
        for file in (self.val_files):
            shutil.copy(file, 'images/val')
            filename = file.split('/')[-1].split('.')[0]
            shutil.copy(os.path.join(label_dir, filename+'.txt'), 'labels/val')

        self.next(self.yolov5)

    @step
    def yolov5(self):

        with open('./train.txt', 'w') as f:
            for path in glob('./images/train/*'):
                f.write(path+'\n')
                    
        with open('./val.txt', 'w') as f:
            for path in glob('./images/val/*'):
                f.write(path+'\n')

        data = dict(
            train =  join( '../vinbigdata/train.txt'),
            val   =  join( '../vinbigdata/val.txt' ),
            nc    = 14,
            names = self.classes
            )

        with open(join( '../yolov5', 'vinbigdata.yaml'), 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

        f = open(join( '../yolov5', 'vinbigdata.yaml'), 'r')

        self.next(self.modeltrain)

    @step
    def modeltrain(self):

        os.system('cd .. && cd yolov5 && python train.py --img 128 --batch 16 --epochs 2 --data vinbigdata.yaml --weights yolov5x.pt --cache')
        
        self.next(self.end)

    @step
    def end(self):
        print("")
        

if __name__ == '__main__':
    VinBig()
   