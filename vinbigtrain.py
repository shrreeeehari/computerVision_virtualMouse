#dependencies
import numpy as np, pandas as pd, torch, IProgress
from glob import glob
import pandas as pd
import shutil, os
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import yaml
from sklearn.model_selection import GroupKFold
from tqdm.notebook import tqdm
from sklearn import datasets, model_selection
from sklearn.model_selection import train_test_split
from metaflow import FlowSpec, Flow, step, IncludeFile, card, Parameter
import os.path
from os.path import exists


class VinBig(FlowSpec):
    data = IncludeFile("data", default="./train.csv")
    exclude_nb_input = Parameter('exclude_nb_input', default=True, type=bool)
    @card(type='notebook')
    @step
    def start(self):
        self.nb_options_dict = dict(input_path='./VinBigTrain.ipynb', exclude_input=self.exclude_nb_input)
        
        self.next(self.end)

    @step
    def end(self):
        print("")
        

if __name__ == '__main__':
    VinBig()
   