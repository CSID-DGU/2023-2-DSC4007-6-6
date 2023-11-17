import torch
import torch.nn as nn
import timm
from config import Config





class ResNextModel(nn.Module):
    def __init__(self):
        super(ResNextModel,self).__init__()
        self.backbone=timm.create_model(Config['FEATURE_EXTRACTOR'],pretrained=True,in_chans=1)
    def forward(self,x):
        return self.backbone(x)
    

class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()
    def forward(self,x):
        return x

class RSNAModel(nn.Module):
    def __init__(self, pretrained=True):
        super(RSNAModel, self).__init__()
        self.backbone = ResNextModel()
        num_features = self.backbone.backbone.fc.in_features
        
        self.backbone.backbone.fc = Identity()
        self.dropout= nn.Dropout(Config['DR_RATE'])
        self.rnn = nn.LSTM(num_features, Config['RNN_HIDDEN_SIZE'], Config['RNN_LAYERS'])
        self.fc1 = nn.Linear(Config['RNN_HIDDEN_SIZE'], Config['NUM_CLASSES'])
        
    def forward(self, x):
        b_z, fr, h, w = x.shape
        ii = 0
        in_pass = x[:, ii].unsqueeze(1)
        y = self.backbone((in_pass))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, fr):
            y = self.backbone((x[:, ii].unsqueeze(1)))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:, -1])
        out = self.fc1(out)
        return out