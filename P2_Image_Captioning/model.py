import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = F.relu(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,max_seq_length=20, num_layers=2):
        """Set the hyper-parameters and build the layers.""" 
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
         
    def forward(self, features, captions):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, (h,c) = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        
        # outputs - batch_size * vocabb_size - 10* 8850
        return outputs
        #lstm_out, _ = self.lstm(inputs)
        #outputs = self.hidden2vocab(lstm_out)
        #allscores = F.softmax(outputs, dim=1)
        #return allscores[:,:-1,:] 
        
    def sample(self, features,states=None,max_seq_length=20):
        """accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)"""
        sampled_ids = []
        #inputs = features.unsqueeze(1)
        inputs = features
        for i in range(max_seq_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.cat(sampled_ids, 0)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids.squeeze()