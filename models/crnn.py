import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 3


class CRNNModel(nn.Module):
    def __init__(self, h1: int = 256, nl: int = 2):
        super().__init__()
        b0   = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        fd   = b0.fc.in_features
        self.cnn  = nn.Sequential(*list(b0.children())[:-1])
        self.lstm = nn.LSTM(
            input_size=fd, hidden_size=h1,
            num_layers=nl, batch_first=True,
            bidirectional=True, dropout=0.3,
        )
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(h1 * 2, NUM_CLASSES),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        f  = self.cnn(x.view(B * T, C, H, W)).squeeze(-1).squeeze(-1)
        f  = f.view(B, T, -1)
        o1, _ = self.lstm(f)
        return self.head(o1[:, -1, :])
