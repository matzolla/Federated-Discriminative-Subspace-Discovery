import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import get_graph_node_names
import math


##### a normal Linear classifier (used for vanilla, our approach and model quantization)
class LinearClassifier(nn.Module):
    def __init__(self,in_dim,num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim,num_classes)
    def forward(self,x):
        return self.fc(x)

#### a one hidden layer classifier

class OneHiddenLayerClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, activation=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

#### a 2 hidden layer classifier
class TwoHiddenLayerClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, num_classes, activation=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim1),
            activation(),
            nn.Linear(hidden_dim1, hidden_dim2),
            activation(),
            nn.Linear(hidden_dim2, num_classes),
        )

    def forward(self, x):
        return self.net(x)
    
#### we use a low-rank approximation of the LinearClassifier, to mimic FLOCORA Implementation

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        assert isinstance(linear, nn.Linear)

        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.A = nn.Parameter(torch.zeros(r, linear.in_features))
        self.B = nn.Parameter(torch.zeros(linear.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base = self.linear(x)
        return base + self.scaling * (self.dropout(x) @ self.A.T @ self.B.T)
    
### we use multi-scale vision transformer for embedding extraction
def video_Mvit():
    model= torchvision.models.video.mvit_v1_b(pretrained=True)
    num_ftrs=model.head[1].in_features
    model.head[1]=nn.Linear(num_ftrs,101)
    return model


class resnet3D(nn.Module):
    def __init__(self, num_classes):
        super(resnet3D, self).__init__()

        # Define R3D_18 as a submodule
        self.r3d_18 = torchvision.models.video.r3d_18(weights='R3D_18_Weights.KINETICS400_V1')

        # Modify the final fully connected layer for the desired number of classes
        self.r3d_18.fc = nn.Linear(self.r3d_18.fc.in_features, num_classes)

    def forward(self, x):
        # Forward pass through R3D_18 model
        x = self.r3d_18(x)

        return x
    
#print(resnet3D(101))
#print(get_graph_node_names(resnet3D(101)))