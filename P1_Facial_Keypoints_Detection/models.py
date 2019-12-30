## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):
     def __init__(self):
        super(Net, self).__init__()
        
         ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
      
        ######### CONV 1 , MAX POOL
        # 1 input image channel (grayscale), 96 output channels/feature maps
        # 11x11 square convolution kernel
        ## 96dimensions: (96, 55, 55)
        # Output size = (W-F)/S+1 = (227-11/4)+1
        # after one pool layer, this becomes (96, , 13)
        self.conv1 = nn.Conv2d(1, 96, 11,stride=4,padding=0)
        

        # maxpool layer
        # pool with kernel_size=2, stride=2
        # after one pool layer, this becomes (96, 27, 27)
        
        self.pool = nn.MaxPool2d(3,2)
        
        ######### CONV 2 , MAX POOL

        # second conv layer: 96 inputs, 27 outputs, 5*5 conv
        ## output size = (W-F)/S +1 = (27-5+2*2)/1 +1 = 27
        # the output tensor will have dimensions: (96,27, 27)
        # after another pool layer this becomes (96, 13, 13)
        self.conv2 = nn.Conv2d(96,256,5,stride=1,padding=2)

        # after another pool layer this becomes (256, 13, 13)
        # Calculate output size 
        self.pool = nn.MaxPool2d(3,2)

        ######### CONV 3 
        # Output =( 13 - 3 + 2*1 )/1 + 1 =  13
        # Output tensor (384,13,13)
        self.conv3 = nn.Conv2d(256,384,3,stride=1,padding=1)


        ######### CONV 4
        #op = (13-3+2*1)/1 + 1 = 13
        #op tensor = (256,13,13)
        self.conv4 = nn.Conv2d(384,256,3,stride = 1,padding=1)


        ########## CONV 5 
        # op ?
        self.conv5 = nn.Conv2d(256,256,3,stride = 1,padding=1)


        # op 128
        self.pool = nn.MaxPool2d(3,2)
        

        self.pool2 = nn.MaxPool2d(2,2)
        



        #20 outputs * the 5*5 filtered/pooled map size
        # 10 output channels (for the 10 classes)
        self.fc1 = nn.Linear(256*6*6,
                             4096)
        
         # dropout with p=0.4
        self.fc1_drop = nn.Dropout2d(p=0.4)
        
        
        self.fc2 = nn.Linear(4096,4096)
        

        # for output classes
        self.fc3 = nn.Linear(4096,136)

        
    

    # define the feedforward behavior
     def forward(self, x):
        # two conv/relu + pool layers
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(F.relu(self.conv5(x)))

        x = self.fc1_drop(x)
        # prep for linear layer
        # flatten the inputs into a vector
        #print(x.size(0))
        x = x.view(x.size(0), -1)
        #x = x.view(256*6*6, -1)


        
        #(_, C, H, W) = x.data.size()
        #x = x.view( -1 , C * H * W)
        
        # one linear layer
        #print(x.size(1), self.fc1.weight.size(1))
        x = F.relu(self.fc1(x))
        # a softmax layer to convert the 10 outputs into a distribution of class scores

        # dropout with p=0.4
        x = self.fc1_drop(x)
        
        
        # 2nd linear layer
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        
        # final output
        return x

      
   
