import os
import torch
import json
import random
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict


class Projector(nn.Module):
    """
    This layer takes fixed-sized embedding(s) named 
    'sentence_embedding' (from the student model) and/or 'source_embedding' (from the teacher model) 
    as inputs, and outputs 'student_hidden_states' and/or 'teacher_hidden_states' respectively.

    Pipeline:
    1) applying L2 normalization, 
    2) (optional) adding gaussian noises and again applying L2 normalization,
    3) applying the feed-forward process in the order of `Linear-Dropout-LayerNorm` 

    :param in_features: Size of the input dimension
    :param out_features: Output size
    :param bias: Add a bias vector
    :param dropout: Probability of dropout (default to 0.1)
    :param noise_std: Standard deviation of the gaussian noise (defaut to 0)
    :param noise_prob: Probability to add gaussian noise (default to 0, which is equivalent to 1)
    :param student_emb_keyname: Features of the specific key to be mapped to `student_hidden_states`
    :param teacher_emb_keyname: Features of the specific key to be mapped to `teacher_hidden_states`
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, bias: bool = True, 
                 dropout: float = 0.1, 
                 noise_std: float = 0.0, 
                 noise_prob: float = 0.0, 
                 student_emb_keyname: str = 'sentence_embedding',
                 teacher_emb_keyname: str = 'source_embedding',
                 **kwargs
                 ):
        super(Projector, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        self.dropout = dropout

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)

        self.student_emb_keyname = student_emb_keyname
        self.teacher_emb_keyname = teacher_emb_keyname

    def forward(self, features: Dict[str, Tensor]):
        for src_name, trg_name in zip(
                [self.student_emb_keyname, self.teacher_emb_keyname], 
                ['student_hidden_states', 'teacher_hidden_states']
            ):
            if src_name in features and features[src_name] is not None:
                # 1): L2 normalization
                feats = F.normalize(features[src_name], dim=-1)
                
                # 2): Gaussian noise & L2 normalization
                if self.noise_std > 0 and self.training:
                    if self.noise_prob == 0 or (random.random() < self.noise_prob):
                        feats = feats + (torch.randn(feats.shape).to(feats.device) * self.noise_std)
                        feats = F.normalize(feats, dim=-1)
                
                # 3): Feed forward in the order of `Linear-Dropout-LayerNorm`
                feats = self.norm(self.drop(self.linear(feats)))

                if feats.dim() == 2:
                    feats = feats.unsqueeze(1) # (batch_size, 1, out_features)
            
                features[trg_name] = feats
        
        return features

    def get_config_dict(self):
        return {
            'in_features': self.in_features, 
            'out_features': self.out_features, 
            'bias': self.bias, 
            'noise_std': self.noise_std, 
            'dropout': self.dropout, 
            'noise_prob': self.noise_prob,
            'student_emb_keyname': self.student_emb_keyname,
            'teacher_emb_keyname': self.teacher_emb_keyname,
        }

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    def __repr__(self):
        return "Projector({})".format(self.get_config_dict())

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        model = Projector(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        return model
