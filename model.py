import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
from torchvision.models.densenet import DenseNet121_Weights

class DenseNet121(models.DenseNet):
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        self.model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.model.features.add_module("relu", nn.ReLU(inplace=True))
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(num_ftrs, num_classes)
        )
        
    def forward(self, x):
        out = self.model.features(x)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = torch.flatten(out, 1)
        out = self.model.classifier(out)
        return out
        
        
class AttDiagnosisModel(nn.Module):
    def __init__(self, num_classes, model_type='resnet50', dropout=True):
        super(AttDiagnosisModel, self).__init__()
        
        if "resnet" in model_type:
            if model_type == "resnet34":
                model_cls = models.resnet34
                model_weights = ResNet34_Weights.IMAGENET1K_V1
            elif model_type == "resnet50":
                model_cls = models.resnet50
                model_weights = ResNet50_Weights.IMAGENET1K_V1
            elif model_type == "resnet101":
                model_cls = models.resnet101
                model_weights = ResNet101_Weights.IMAGENET1K_V2
            grad_layer = "layer4"
            self.model = model_cls(weights=model_weights)
            num_ftrs = self.model.fc.in_features
            if dropout:
                self.model.fc = nn.Sequential(
                    nn.Dropout(0.25),
                    nn.Linear(num_ftrs, num_classes)
                )
            else:
                self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_type == "densenet121":
            grad_layer = "model.features"
            self.model = DenseNet121(num_classes)
        self.feed_forward_features = None
        self.backward_features = None
        self._register_hooks(grad_layer)
        self.omega = 100 # steep sigmoid
        self.sigma = 0.25 # threshold

    def _register_hooks(self, grad_layer):
        def forward_hook(module, grad_input, grad_output): # after forward
            self.feed_forward_features = grad_output
        def backward_hook(module, grad_input, grad_output): # every gradient comput for modules
            self.backward_features = grad_output[0]
        for name, m in self.model.named_modules():
            if name == grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_full_backward_hook(backward_hook)
                
    def calc_feature(self, x):
        layers = list(self.model.children())
        model_before_fc = nn.Sequential(*layers[:-1]) # except the last layer
        feats = model_before_fc(x)
        return feats

    def forward(self, x, labels):
        self.model.zero_grad()
        with torch.enable_grad():
            if self.model.training:
                logits = self.model(x)  # B x num_classes
                grad = logits.gather(1, labels.long().view(-1,1))
                grad_logits = grad.sum()  # B x num_classes
            else:
                self.model.eval()
                logits = self.model(x)  # B x num_classes
                pred = F.softmax(logits, dim=1).argmax(dim=1)
                grad = logits.gather(1, pred.long().view(-1,1))
                grad_logits = grad.sum()  # B x num_classes
        grad_logits.backward(gradient=grad_logits, retain_graph=True)
        w = F.adaptive_avg_pool2d(self.backward_features, 1)
        att = torch.mul(self.feed_forward_features, w).sum(dim=1, keepdim=True)
        att = F.relu(att)
        att = F.interpolate(att, size=x.size()[2:], mode='bilinear', align_corners=False)
        att_scaled = (att - att.min()) / (att.max() - att.min()) # -> [0, 1]
        att_mask = torch.sigmoid(self.omega * (att_scaled - self.sigma))
        return logits, att_mask, att
