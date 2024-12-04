import torch
import torch.nn as nn
from networks import get_AlexNet

import os

def get_new_task_classifier(class_num, prev_model_path = None, head_shared = False):

    if prev_model_path is not None:
        model = get_AlexNet()
        model.load_state_dict(torch.load(prev_model_path))
    else: 
        model = get_AlexNet(True)

    if not head_shared:
        last_layer_index = str(len(model.classifier._modules) - 1)
        num_ftrs = model.classifier._modules[last_layer_index].in_features
        model.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(class_num))
        print("NEW FC CLASSIFIER HEAD with {} units".format(len(class_num)))

    return model

def load_classifier(out_dim, model_path = None):

    model = get_AlexNet()
    last_layer_index = str(len(model.classifier._modules) - 1)
    num_ftrs = model.classifier._modules[last_layer_index].in_features
    model.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(out_dim))
    model.load_state_dict(torch.load(model_path))

    return model