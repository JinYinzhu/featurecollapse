import numpy as np
import torch
import os

class Ellipse():
    def __init__(self, data_path) -> None:
        circle = np.load('./data/X_0.npy')
        line = np.load('./data/X_1.npy')
        X = np.concatenate([circle,line],0)