"""Models for facial keypoint detection"""

from re import L
from telnetlib import EL
import torch
import torch.nn as nn

class KeypointModel(nn.Module):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        """
        super().__init__()
        self.hparams = hparams
        self.device = hparams["device"]

        # in_channels =  number of input channels (or depth) of the input volume.
        
        # out_channels = number of output channels produced by the convolutional layer. 
        # Each output channel corresponds to a convolutional kernel (filter) that will be applied to the input
        # Essentially, it determines the number of filters used by the convolutional layer.
        
        # kernel_size = Specifies the size of the convolutional kernel
        
        # padding = Padding is used to add extra pixels (usually zero-valued) around the input data.
        
        # stride = Stride determines the step size the convolutional kernel takes as it slides/spatially moves across the input


        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, padding=0, stride=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0, stride=2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=0, stride=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(128*5*5, 1000),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 1000),
            # Linear activation
            nn.Dropout(0.6),
            nn.Linear(1000, 30)
        )

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        #x = x.to(self.device)

        # Move the model to the same device as the input data
        #self.model = self.model.to(x.device)

        x = self.model(x)

        return x

class DummyKeypointModel(nn.Module):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
