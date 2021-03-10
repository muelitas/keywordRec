'''/**************************************************************************
    File: models.py
    Author(s): Mario Esparza, Luis Sanchez
    Date: 02/26/2021
    
    TODO
    
***************************************************************************''' 
#Source: https://colab.research.google.com/drive/1IPpwx4rX32rqHKpLz7dc8sOKspUa-YKO#scrollTo=RVJs4Bk8FjjO
import torch.nn as nn
import torch.nn.functional as F

class BiGRU(nn.Module):

    def __init__(self, gru_dim, gru_hid_dim, gru_layers, dropout, batch_first):
        super(BiGRU, self).__init__()

        self.BiGRU = nn.GRU(input_size=gru_dim, hidden_size=gru_hid_dim,
            num_layers=gru_layers, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(gru_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):
    def __init__(self, hparams):        
        super(SpeechRecognitionModel, self).__init__()
        #If we need to implement a second CNN, use this:
        #https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
        self.cnn1_kernel = hparams['cnn1_kernel']
        self.cnn1_stride = hparams['cnn1_stride']
        
        # cnn for extracting heirachal features
        self.cnn = nn.Conv2d(1, hparams['cnn1_filters'], hparams['cnn1_kernel'], 
            stride=hparams['cnn1_stride'])
                               
        in_feats = hparams['cnn1_filters'] * (hparams['n_mels']//hparams['cnn1_stride'])
        self.fully_connected = nn.Linear(in_feats, hparams['gru_dim'])
        
        self.birnn_layers = BiGRU(hparams['gru_dim'], hparams['gru_hid_dim'],
            hparams['gru_layers'], hparams['gru_dropout'], batch_first=True)

        #Dynamiclly, set up value for classifier's out-ftrs of fc layers
        fc_out_ftrs = 0
        
        if hparams['gru_hid_dim']*2 <= hparams['n_class']:
            fc_out_ftrs = hparams['n_class']
        else:
            fc_out_ftrs = (hparams['gru_hid_dim']*2 - hparams['n_class']) // 2
            fc_out_ftrs += hparams['n_class']
        
        self.classifier = nn.Sequential(
            nn.Linear(hparams['gru_hid_dim']*2, fc_out_ftrs),  # birnn returns gru_hid_dim*2 (2 because bidirectional=True in nn.GRU)
            nn.GELU(),
            nn.Dropout(hparams['dropout']),
            nn.Linear(fc_out_ftrs, hparams['n_class'])
        )

    def add_paddings(self, x):
        #Pad if necessary
        paddings = [0,0,0,0] #left, right, top, bottom
        sizes = x.size()
        if (sizes[2] % 2) != 0: #H
            paddings[3] += 1
            
        if (sizes[3] % 2) != 0: #W
            paddings[1] += 1
        
        #Equation below won't work if cnn1_stride != 2
        pad_val = int(self.cnn1_kernel/self.cnn1_stride - 0.5)
        paddings = [val + pad_val for val in paddings]
        self.pad = nn.ZeroPad2d(paddings)

    def forward(self, x):
        x = self.pad(x)
        x = self.cnn(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x