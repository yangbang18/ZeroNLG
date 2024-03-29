import os
import json
import torch
from torch import Tensor
from torch import nn
from typing import Dict
from ..utils import fullname, import_from_string


# Derived from sentence_transformers.models.Dense
# (https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Dense.py)
class Dense(nn.Module):
    """Feed-forward function with  activiation function.

    This layer takes a fixed-sized sentence embedding and passes it through a feed-forward layer. Can be used to generate deep averaging networks (DAN).

    :param in_features: Size of the input dimension
    :param out_features: Output size
    :param bias: Add a bias vector
    :param activation_function: Pytorch activation function applied on output
    :param init_weight: Initial value for the matrix of the linear layer
    :param init_bias: Initial value for the bias of the linear layer
    :param proj_token_embs: If True, project token embeddings in addition to the sentence embedding
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True, 
                 activation_function=nn.Tanh(), 
                 init_weight: Tensor = None, 
                 init_bias: Tensor = None,
                 proj_token_embs: bool = False,
                 **kwargs,
                ):
        """
        Yang B. modification:
        1) add a new parameter `proj_token_embs`, which allow projecting token embeddings if speficied
        2) save `proj_token_embs` to config_dict
        """
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation_function = activation_function
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if init_weight is not None:
            self.linear.weight = nn.Parameter(init_weight)

        if init_bias is not None:
            self.linear.bias = nn.Parameter(init_bias)
        
        self.proj_token_embs = proj_token_embs

    def forward(self, features: Dict[str, Tensor]):
        features.update({'sentence_embedding': self.activation_function(self.linear(features['sentence_embedding']))})
        if self.proj_token_embs:
            features.update({'token_embeddings': self.activation_function(self.linear(features['token_embeddings']))})
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.out_features

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    def get_config_dict(self):
        return {'in_features': self.in_features, 'out_features': self.out_features, 'bias': self.bias, 'activation_function': fullname(self.activation_function), 'proj_token_embs': self.proj_token_embs}
    
    def __repr__(self):
        return "Dense({})".format(self.get_config_dict())

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        config['activation_function'] = import_from_string(config['activation_function'])()
        model = Dense(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        return model
