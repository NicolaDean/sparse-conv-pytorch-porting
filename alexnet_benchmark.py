import sparse_conv as sp

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import copy



class AlexNet(sp.SparseModel):
    def __init__(self, n_classes,sparse_conv_flag=True):
        self._sparse_conv_flag=sparse_conv_flag
        super(AlexNet, self).__init__(sparse_conv_flag)

        self.layer1 = nn.Sequential(
            self.conv(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            self.conv(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            self.conv(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            self.conv(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            self.conv(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, n_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

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
    

N_CLASSES       = 10
IMG_SIZE        = 227
BATCH_SIZE      = 32
INPUT_CHANNELS  = 3
PRUNING_PARAMETER = 0.90

INPUT_SHAPE = (BATCH_SIZE,INPUT_CHANNELS,IMG_SIZE,IMG_SIZE)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

#LOAD THE MODEL
model = AlexNet(N_CLASSES,sparse_conv_flag=True)
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
model._initialize_sparse_layers(input_shape=INPUT_SHAPE)
model._set_sparse_layers_mode(sp.Sparse_modes.Calibration)

#------------------------------------------
#------------------------------------------
#----------TESTING-------------------------
#------------------------------------------
#------------------------------------------

#Generate a dummy input to give the convolution

print("----------------------------------")
print("-----Example of Benchmark or Test-------")
print("----------------------------------")

dummy_input = torch.randn(INPUT_SHAPE, dtype=torch.float).to(device)
dummy_input = dummy_input.cuda()

print(f"INPUT SHAPE : {dummy_input.shape}")
input = copy.deepcopy(dummy_input)
input = input.cuda()

model.forward(dummy_input)

exit()