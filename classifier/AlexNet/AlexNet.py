import torch
import torch.nn as nn
from classifier.AlexNet.networks import get_AlexNet

import os

def get_new_task_classifier(out_dim, prev_out_dim = 100, prev_model_path = None, head_shared = False, device = 'cpu'):

    if prev_model_path is not None:
        model = get_AlexNet()
        last_layer_index = str(len(model.classifier._modules) - 1)
        num_ftrs = model.classifier._modules[last_layer_index].in_features
        model.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, prev_out_dim)
        model.load_state_dict(torch.load(prev_model_path))
    else: 
        model = get_AlexNet(True)

    if not head_shared:
        last_layer_index = str(len(model.classifier._modules) - 1)
        num_ftrs = model.classifier._modules[last_layer_index].in_features
        model.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, out_dim)
        print("NEW FC CLASSIFIER HEAD with {} units".format(out_dim))

    model.to(device)

    return model

def load_classifier(out_dim, model_path = None):

    model = get_AlexNet()
    last_layer_index = str(len(model.classifier._modules) - 1)
    num_ftrs = model.classifier._modules[last_layer_index].in_features
    model.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(out_dim))
    model.load_state_dict(torch.load(model_path))

    return model