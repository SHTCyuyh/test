from turtle import forward
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class encoder(nn.Module):
    def __init__(self, in_features = 512, embedding_size = 2048, hidden_features = 4096, n_layers = 4, dropout = 0.):
        super().__init__()
        self.hidden_feature = hidden_features
        self.num_layers = n_layers
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        # self.rnn = nn.LSTM(embedding_size, hidden_features, n_layers, dropout = dropout)
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        hidden = self.drop(F.relu(self.fc2(self.fc1(x))))
        

        # out, (hidden, cell) = self.rnn(embed)/

        return hidden



if __name__== '__main__':
    import os
    import numpy as np
    from einops import rearrange

    path1 = '/data2/og_data/person18/meas/person18-00000.hdr'
    path2 = '/data2/og_data/person18/meas/person18-00001.hdr'
    data1 =  cv2.imread(path1, -1)
    data1 = data1/np.max(data1)
    data1 =cv2.cvtColor(data1, cv2.COLOR_BGR2GRAY)
    data1 = data1/np.max(data1)
    data1 = rearrange(data1, '(t h) w ->h w t', t=600)[:,:,:512]

    data2 =  cv2.imread(path2, -1)
    data2 = data2/np.max(data2)
    data2 =cv2.cvtColor(data2, cv2.COLOR_BGR2GRAY)
    data2 = data2/np.max(data2)
    data2 = rearrange(data2, '(t h) w ->h w t', t=600)[:,:,:512]


    model = encoder()

    data1 = torch.from_numpy(data1)
    data2 = torch.from_numpy(data2)


    hideen1 = model(data1)
    hideen2 = model(data2)

    print('done')


