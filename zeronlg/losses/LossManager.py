import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Iterable, Dict, Tuple
from ..Framework import Framework


class LossManager(nn.Module):
    def __init__(self, 
                 model: Framework, 
                 loss_mse_scale: float = 1.0, 
                 loss_at_teacher_scale: float = 0.0, 
                 loss_at_student_scale: float = 0.0, 
                 loss_contrastive_scale: float = 0.0,
                 ):
        """
        :param model: Framework based on sentence-transformers
        :loss_mse_scale: The scale of MSE loss between teacher's and student's sentence embeddings
        :loss_at_teacher_scale: The scale of cross-entropy loss of decoding on teacher's sentence embeddings (translation)
        :loss_at_teacher_scale: The scale of cross-entropy loss of decoding on student's sentence embeddings (auto-encoding)
        :loss_contrastive_scale: The scale of contrastive loss between teacher's and student's sentence embeddings
        """
        super(LossManager, self).__init__()
        self.model = model

        self.loss_fct = nn.MSELoss()

        self.loss_mse_scale = loss_mse_scale
        self.loss_at_teacher_scale = loss_at_teacher_scale
        self.loss_at_student_scale = loss_at_student_scale
        self.loss_contrastive_scale = loss_contrastive_scale

        if loss_contrastive_scale > 0:
            self.temp = nn.Parameter(torch.ones([]) * 0.07) # identical to CLIP
    
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        outputs = self.model(sentence_features)

        loss, loss_msg_dict = 0, {}
        for name in ['mse', 'at_teacher', 'at_student', 'contrastive']:
            this_loss, this_dict = getattr(self, f'forward_{name}')(outputs, labels)
            loss += this_loss
            loss_msg_dict.update(this_dict)

        return loss, loss_msg_dict

    def forward_mse(self, 
            outputs: Dict[str, Tensor], 
            labels: Tensor,
            name: str = 'loss_mse'
            ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Computes the MSE loss between the computed sentence embedding and a target sentence embedding. This loss
        is used when extending sentence embeddings to new languages as described in our publication
        Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation: https://arxiv.org/abs/2004.09813

        For an example, see the documentation on extending language models to new languages.
        """
        if self.loss_mse_scale > 0:
            loss = self.loss_fct(outputs['sentence_embedding'], labels)
            return self.loss_mse_scale * loss, {name: loss.detach().cpu().item()}
        
        return 0, {}

    def forward_at_teacher(self, 
            outputs: Dict[str, Tensor], 
            labels: Tensor,
            name: str = 'loss_at_teacher',
            ) -> Tuple[Tensor, Dict[str, float]]:
        
        if self.loss_at_teacher_scale > 0:
            loss = outputs['loss_at_teacher']
            return self.loss_at_teacher_scale * loss, {name: loss.detach().cpu().item()}
        
        return 0, {}

    def forward_at_student(self,
            outputs: Dict[str, Tensor], 
            labels: Tensor,
            name: str = 'loss_at_student',
            ) -> Tuple[Tensor, Dict[str, float]]:

        if self.loss_at_student_scale > 0:
            loss = outputs['loss_at_student']
            return self.loss_at_student_scale * loss, {name: loss.detach().cpu().item()}
        
        return 0, {}

    def forward_contrastive(self,
            outputs: Dict[str, Tensor], 
            labels: Tensor,
            name: str = 'loss_cl'
            ) -> Tuple[Tensor, Dict[str, float]]:

        if self.loss_contrastive_scale > 0:
            with torch.no_grad():
                self.temp.clamp_(0.001, 0.5)
        
            feats_student = F.normalize(outputs['sentence_embedding'], dim=-1)
            feats_teacher = F.normalize(labels, dim=-1)
            logits_s2t = feats_student @ feats_teacher.t() / self.temp
            logits_t2s = feats_teacher @ feats_student.t() / self.temp
            
            cl_labels = torch.arange(logits_s2t.size(0), device=logits_s2t.device)

            loss_s2t = F.cross_entropy(logits_s2t, cl_labels, reduction='mean')
            loss_t2s = F.cross_entropy(logits_t2s, cl_labels, reduction='mean')
            loss = (loss_s2t + loss_t2s) / 2
            
            return self.loss_contrastive_scale * loss, {
                name: loss.detach().cpu().item(),
                'temp': self.temp.detach().cpu().item(),
            }
        
        return 0, {}
