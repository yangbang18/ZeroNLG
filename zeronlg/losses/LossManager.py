from torch import nn, Tensor
from typing import Iterable, Dict, Tuple
from ..Framework import Framework


class LossManager(nn.Module):
    def __init__(self, 
                 model: Framework, 
                 loss_mse_scale: float = 1.0, 
                 loss_at_teacher_scale: float = 0.0, 
                 loss_at_student_scale: float = 0.0, 
                 ):
        """
        :param model: Framework based on sentence-transformers
        :loss_mse_scale: The scale of MSE loss between teacher's and student's sentence embeddings
        :loss_at_teacher_scale: The scale of cross-entropy loss of decoding on teacher's sentence embeddings (translation)
        :loss_at_teacher_scale: The scale of cross-entropy loss of decoding on student's sentence embeddings (auto-encoding)
        """
        super(LossManager, self).__init__()
        self.model = model

        self.loss_fct = nn.MSELoss()

        self.loss_mse_scale = loss_mse_scale
        self.loss_at_teacher_scale = loss_at_teacher_scale
        self.loss_at_student_scale = loss_at_student_scale
    
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        outputs = self.model(sentence_features)

        loss, loss_msg_dict = 0, {}
        for name in ['mse', 'at_teacher', 'at_student']:
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
