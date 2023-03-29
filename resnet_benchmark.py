import sparse_conv as sp

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import copy


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        sp.SparseConv2D(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        #nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        sp.SparseConv2D(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        #nn.BatchNorm2d(out_channels)
                        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(sp.SparseModel):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                sp.SparseConv2D(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def pruning_model_random(model, px):

    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, sp.SparseConv2D):
            print(f"Pruning layer {name}")
            parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    ) 
    
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 4
PRUNING_PARAMETER = 0.7
IMG_SIZE = 224
N_CLASSES = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)
model.to(device)

#PRUNE THE MODEL TO ADD SPARSITY
print("--------------------------------------")
print(f"-----Pruning the Network at [{PRUNING_PARAMETER}]-----")
print("--------------------------------------")
pruning_model_random(model,PRUNING_PARAMETER)

#SET MODEL IN TESTING MODE (For each SparseConv compare Conv2D with SparseConv2D)
print("----------------------------------")
print("-----Initialize the Network-------")
print("----------------------------------")
model._initialize_sparse_layers(input_shape=(32,3,IMG_SIZE,IMG_SIZE))
model._set_sparse_layers_mode(sp.Sparse_modes.Benchmark)

#------------------------------------------
#------------------------------------------
#----------TESTING-------------------------
#------------------------------------------
#------------------------------------------

#Generate a dummy input to give the convolution

print("----------------------------------")
print("-----Example of Benchmark or Test-------")
print("----------------------------------")
batch_size = 32
dummy_input = torch.randn(batch_size, 3,IMG_SIZE,IMG_SIZE, dtype=torch.float).to(device)
dummy_input = dummy_input.cuda()

print(f"INPUT SHAPE : {dummy_input.shape}")
input = copy.deepcopy(dummy_input)
input = input.cuda()

model.forward(dummy_input)

exit()