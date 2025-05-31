import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from data.dataset import BCDataset
idata = BCDataset(type='ibm', path='datasets/ibm')
print(idata.features.shape, idata.labels.shape, idata.edge_index.shape)