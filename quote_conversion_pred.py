# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:28:32 2019

@author: shourt
"""

import pandas as pd

dataset = pd.read_csv('dataset_quote.csv')
X= dataset.iloc[:, :-1].values
y= dataset.iloc[:, -1].values