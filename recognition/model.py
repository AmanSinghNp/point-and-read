import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256, dropout=0.3):
        super(CRNN, self).__init__()
        
        # 1. CNN Backbone (ResNet-34)
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        # Modify the first conv layer to accept 1-channel grayscale input instead of 3 RGB channels
        # ResNet's first conv layer: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = resnet.conv1.weight.data.sum(dim=1, keepdim=True) # Sum weights across RGB channels
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # In a standard ResNet, layer4 output shape is (batch, 512, H/32, W/32)
        # We need to pool aggressively in height, conservatively in width.
        # So we add a custom pooling layer to squash the height to 1 while preserving the width sequence.
        # Assuming input H=64, after layer4 we are at H=2.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None)) 
        
        # 2. BiLSTM Layers
        # Input size is 512 (channels from ResNet layer4)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=False # (Sequence, Batch, Features)
        )
        
        # 3. Linear Projection
        # 2 directions * hidden_size (256) = 512
        self.fc = nn.Linear(hidden_size * 2, num_classes + 1) # +1 for CTC blank token

    def forward(self, x):
        # x shape: (batch_size, 1, 64, W)
        
        # CNN Feature Extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x shape: (batch_size, 512, H_feat, W_feat)
        
        x = self.adaptive_pool(x)
        # x shape: (batch_size, 512, 1, W_feat)
        
        # Reshape for RNN: (batch_size, channels, 1, width) -> (batch_size, channels, width)
        x = x.squeeze(2)
        
        # Map to Sequence: (batch_size, channels, width) -> (width, batch_size, channels)
        x = x.permute(2, 0, 1)
        
        # BiLSTM
        x, _ = self.rnn(x)
        # x shape: (width, batch_size, hidden_size * 2)
        
        # Linear Projection
        x = self.fc(x)
        # x shape: (width, batch_size, num_classes + 1)
        
        # Apply log_softmax for CTC loss
        x = nn.functional.log_softmax(x, dim=2)
        
        return x

if __name__ == "__main__":
    # Test the model with dummy data
    num_classes = 79 # Example for IAM dataset
    model = CRNN(num_classes=num_classes)
    
    # Dummy input: batch_size=2, channels=1 (grayscale), height=64, width=256
    dummy_input = torch.randn(2, 1, 64, 256)
    
    print(f"Input shape: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"Output shape: {output.shape} -> (Sequence Length, Batch Size, Num Classes + 1)")
