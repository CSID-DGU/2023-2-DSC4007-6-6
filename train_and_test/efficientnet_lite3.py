import torch
import torch.nn as nn
import timm
from config import Config



class EfficientNetLite3Model(nn.Module):
    def __init__(self):
        super(EfficientNetLite3Model, self).__init__()
        # Initialize the EfficientNet model
        self.backbone = timm.create_model('tf_efficientnet_lite3', pretrained=True, in_chans=1)

    def forward(self, x):
        return self.backbone(x)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class ESNAModel(nn.Module):
    def __init__(self, pretrained=True):
        super(ESNAModel, self).__init__()
        self.backbone = EfficientNetLite3Model()
       
        if isinstance(self.backbone.backbone.classifier, nn.Sequential):
            num_features = self.backbone.backbone.classifier[-1].in_features
        elif isinstance(self.backbone.backbone.classifier, nn.Linear):
            
            num_features = self.backbone.backbone.classifier.in_features
        else:
            raise TypeError("Unsupported classifier type in EfficientNetLite3 model")
        
        
        self.backbone.backbone.classifier = Identity()
        self.dropout = nn.Dropout(Config['DR_RATE'])
        self.rnn = nn.LSTM(num_features, Config['RNN_HIDDEN_SIZE'], Config['RNN_LAYERS'])
        self.fc1 = nn.Linear(Config['RNN_HIDDEN_SIZE'], Config['NUM_CLASSES'])
        
    def forward(self, x):
        b_z, fr, h, w = x.shape
        in_pass = x[:, 0].unsqueeze(1)
        y = self.backbone(in_pass)
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, fr):
            y = self.backbone(x[:, ii].unsqueeze(1))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:, -1])
        out = self.fc1(out)
        return out
