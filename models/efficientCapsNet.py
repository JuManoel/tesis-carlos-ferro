import torch
import torch.nn as nn
from torch.autograd import Variable

# Creacion de la efficientCapsNet, codigo copido del github
# no fue possible entrenar la capsnet, demasiado pesada

def squash(x, eps=10e-21):
    n = torch.norm(x, dim=-1, keepdim=True)
    return (1 - 1 / (torch.exp(n) + eps)) * (x / (n + eps))

def length(x):
    return torch.sqrt(torch.sum(x ** 2, dim=-1) + 1e-8)

def mask(x):
    if isinstance(x, list):
        x, mask = x
    else:
        lengths = torch.sqrt(torch.sum(x ** 2, dim=2))
        _, max_length_indices = lengths.max(dim=1, keepdim=True)
        mask = F.one_hot(max_length_indices, num_classes=x.size(1)).float()
        mask = mask.squeeze(1)
    masked = x * mask.unsqueeze(2)
    return masked.view(masked.size(0), -1)

class PrimaryCapsLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, num_capsules, dim_capsules, stride=2):
        super(PrimaryCapsLayer, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            padding=0,
        )
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules

    def forward(self, input):
        output = self.depthwise_conv(input)  # (64, 128, 92, 92)
        batch_size = output.size(0)

        num_capsules = self.num_capsules  # 16
        dim_capsules = 8  # Deve ser compatível com RoutingLayer

        output = output.view(batch_size, num_capsules, dim_capsules, -1)
        output = output.permute(0, 1, 3, 2).contiguous()
        output = output.view(batch_size, num_capsules, dim_capsules)

        print(f"After reshape: {output.shape}")  # Deve ser (64, 16, 8)

        return squash(output)

class RoutingLayer(nn.Module):
    def __init__(self, num_capsules, dim_capsules):
        super(RoutingLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(num_capsules, num_capsules, dim_capsules, dim_capsules))
        self.b = nn.Parameter(torch.zeros(num_capsules, num_capsules, 1))
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, input):
        print(f"Input shape: {input.shape}")
        print(f"Weights shape: {self.W.shape}")
        u = torch.einsum(
            "...ji,kjiz->...kjz", input, self.W
        )
        print(f"u shape: {u.shape}")
        c = torch.einsum("...ij,...kj->...i", u, u)[
            ..., None
        ]
        print(f"c shape: {c.shape}")
        c = c / torch.sqrt(
            torch.Tensor([self.dim_capsules]).type(torch.cuda.FloatTensor)
        )
        c = torch.softmax(c, axis=1)
        c = c + self.b
        s = torch.sum(
            torch.mul(u, c), dim=-2
        )
        return squash(s)

class EfficientCapsNet(nn.Module):
    def __init__(self):
        super(EfficientCapsNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5, padding=2
        )
        self.batch_norm1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(128)

        self.primary_caps = PrimaryCapsLayer(
            in_channels=128, kernel_size=9, num_capsules=16, dim_capsules=16
        )
        self.digit_caps = RoutingLayer(num_capsules=10, dim_capsules=8)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)

    def forward(self, x):
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = torch.relu(self.batch_norm3(self.conv3(x)))
        x = torch.relu(self.batch_norm4(self.conv4(x)))
        print(x.shape)
        x = self.primary_caps(x)
        print(x.shape)
        x = self.digit_caps(x)
        probs = length(x)
        return x, probs

class ReconstructionNet(nn.Module):
    def __init__(self, n_dim=16, n_classes=10):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 384 * 384)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        x = mask(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.view(-1, 1, 384, 384)

class EfficientCapsNetWithReconstruction(nn.Module):
    def __init__(self, efficient_capsnet, reconstruction_net):
        super(EfficientCapsNetWithReconstruction, self).__init__()
        self.efficient_capsnet = efficient_capsnet
        self.reconstruction_net = reconstruction_net

    def forward(self, x):
        x, probs = self.efficient_capsnet(x)
        reconstruction = self.reconstruction_net(x)
        return reconstruction, probs

class MarginLoss(nn.Module):
    def __init__(self, m_pos=0.9, m_neg=0.1, lambda_=0.5):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, y_pred, y_true, size_average=True):
        t = torch.zeros(y_pred.size()).long()
        if y_true.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, y_true.data.view(-1, 1), 1)
        targets = Variable(t)
        losses = targets * torch.pow(
            torch.clamp(self.m_pos - y_pred, min=0.0), 2
        ) + self.lambda_ * (1 - targets) * torch.pow(
            torch.clamp(y_pred - self.m_neg, min=0.0), 2
        )
        return losses.mean() if size_average else losses.sum()

def efficientCapsNet():
    model = EfficientCapsNet()
    reconstruction_model = ReconstructionNet(16, 1)
    model = EfficientCapsNetWithReconstruction(model, reconstruction_model)
    return model