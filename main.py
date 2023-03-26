import sparse_conv as sp

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune

class LeNet5(nn.Module):
    """
    A standard LeNet5 model
    """

    def __init__(self, n_classes,sparse_conv_flag=True):
        self._sparse_conv_flag=sparse_conv_flag
        super(LeNet5, self).__init__()

        if sparse_conv_flag:
          conv = sp.SparseConv2D
        else:
          conv = nn.Conv2d

        self.conv1 = conv(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = conv(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = conv(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.tanh3 = nn.Tanh()
        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.tanh4 = nn.Tanh()
        self.linear2 = nn.Linear(in_features=84, out_features=n_classes)
    def make_weights_sparse(self):
      '''
      Allow the convolution to compute the sparse representation of the weights
      '''
      print("----------------------------------------")
      self.conv1.load()
      print("----------------------------------------")
      self.conv2.load()
      print("----------------------------------------")
      self.conv3.load()
      print("----------------------------------------")

    #Required only for benchmark Conv2d vs SparseConv
    def setSparseConvUsage(self,usage=True):
      self.conv1.use_sparse = usage
      self.conv2.use_sparse = usage
      self.conv3.use_sparse = usage

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.tanh3(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.tanh4(x)
        logits = self.linear2(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

def pruning_model_random(model, px):

    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
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

IMG_SIZE = 12
N_CLASSES = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

#LOAD THE MODEL
model = LeNet5(N_CLASSES,sparse_conv_flag=True)
model.to(device)

#PRUNE THE MODEL TO ADD SPARSITY
pruning_model_random(model,0.6)

#CONVERT KERNEL TO CSR
model.conv1.training = False
model.conv1.use_sparse = True

#------------------------------------------
#------------------------------------------
#----------TESTING-------------------------
#------------------------------------------
#------------------------------------------

#Generate a dummy input to give the convolution
dummy_input = torch.randn(1, 1,IMG_SIZE,IMG_SIZE, dtype=torch.float).to(device)
dummy_input = dummy_input.cuda()

#Generate sparse conv ouptput
sp_out = model.conv1.forward(dummy_input)

#Generate vanilla conv output
model.conv1.use_sparse = False
out = model.conv1.forward(dummy_input)

#TODO Compare vanilla vs sparse output

print(f"SP_OUT: {sp_out}")
print(f"OUT: {out}")

print(dummy_input.shape)
print(out.shape)
