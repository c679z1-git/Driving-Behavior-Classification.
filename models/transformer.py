import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 3
NUM_FRAMES  = 32


class TransformerModel(nn.Module):
    def __init__(self, dm: int = 256, nh: int = 4,
                 nl: int = 3, dff: int = 512, do: float = 0.1):
        super().__init__()
        b0   = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        fd   = b0.fc.in_features
        self.cnn  = nn.Sequential(*list(b0.children())[:-1])
        self.proj = nn.Linear(fd, dm)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dm))
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_FRAMES + 1, dm))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        el = nn.TransformerEncoderLayer(
            d_model=dm, nhead=nh, dim_feedforward=dff,
            dropout=do, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(el, num_layers=nl)
        self.norm = nn.LayerNorm(dm)
        self.head = nn.Sequential(
            nn.Dropout(do),
            nn.Linear(dm, NUM_CLASSES),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        f  = self.cnn(x.view(B * T, C, H, W)).squeeze(-1).squeeze(-1)
        f  = self.proj(f).view(B, T, -1)

        ct = self.cls_token.expand(B, -1, -1)
        sq = torch.cat([ct, f], dim=1)
        sq = sq + self.pos_embed

        o1 = self.transformer(sq)
        o1 = self.norm(o1[:, 0])
        return self.head(o1)
