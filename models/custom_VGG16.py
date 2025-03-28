import torch
import torch.nn as nn
import numpy as np

# Creacion de una VGG16 personalizada
# principales cambios: 
"""
1. batchnorm en cada capa convolucional
2. activacion GELU en cada capa convolucional
3. dropout en las capas lineales
4. inicializacion de los pesos de las capas convolucionales y lineales
"""
class PyTorchVGG16Logits(nn.Module):

    def __init__(self, num_outputs=1):
        super().__init__()        
        self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=64,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1), 
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(in_channels=64,
                          out_channels=64,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        self.block_2 = nn.Sequential(
                nn.Conv2d(in_channels=64,
                          out_channels=128,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.Conv2d(in_channels=128,
                          out_channels=128,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_3 = nn.Sequential(        
                nn.Conv2d(in_channels=128,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(256),
                nn.GELU(),
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(256), 
                nn.GELU(),        
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(256),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
          
        self.block_4 = nn.Sequential(   
                nn.Conv2d(in_channels=256,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(512),
                nn.GELU(),        
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(512),
                nn.GELU(),        
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(512),
                nn.GELU(),            
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_5 = nn.Sequential(
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(512),
                nn.GELU(),            
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(512),
                nn.GELU(),            
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.BatchNorm2d(512),
                nn.GELU(),    
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))             
        )
        
        self.features = nn.Sequential(
            self.block_1, self.block_2, 
            self.block_3, self.block_4, 
            self.block_5
        )
            
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12*12*512, 4096),
            nn.BatchNorm1d(4096),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_outputs),
        )
             
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.weight.detach().normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.detach().normal_(0, 0.05)
                m.bias.detach().detach().zero_()    

        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x