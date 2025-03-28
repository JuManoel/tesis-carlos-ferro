import torch
import torch.nn as nn
from copy import deepcopy
class MetaClassifier(nn.Module):
    def __init__(self, models: list[nn.Module], num_classes: int=1 , meta_classifier: nn.Module=None):
        super(MetaClassifier, self).__init__()
        self.models = deepcopy(models)
        if not meta_classifier:
            self.fc = nn.Sequential(
            nn.Linear(len(models), 4096),
            nn.BatchNorm1d(4096),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        else:
            self.fc = meta_classifier

    def forward(self, x):
        device = next(self.fc.parameters()).device
        x = x.to(device)
        list_of_outputs = []
        for model in self.models:
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                output_model = model(x)
            model.to('cpu')
            list_of_outputs.append(output_model)
        x = self.fc(torch.cat(list_of_outputs, dim=1))
        return x