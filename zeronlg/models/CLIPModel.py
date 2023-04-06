import os
import json
import transformers
import torch
from torch import nn
from PIL import Image


# Adapted from sentence_transformers.models.CLIPModel
# (https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/CLIPModel.py)
class CLIPModel(nn.Module):
    def __init__(self,  model_name: str = "openai/clip-vit-base-patch32", processor_name = None, use_clip_tokens: bool = False):
        """
        Yang B. modification: 
        1) add truncation=True and max_length=77 in CLIPModel.tokenize to avoid bugs
        2) add function: get_word_embedding_dimension and get_sentence_embedding_dimension
        3) add an extra argument `use_clip_tokens` to return token-level embeddings in addition to the global-level vector
        4) save configs for re-loading
        """
        super(CLIPModel, self).__init__()
        self.config_keys = ['use_clip_tokens']

        if processor_name is None:
            processor_name = model_name

        self.model = transformers.CLIPModel.from_pretrained(model_name)
        self.processor = transformers.CLIPProcessor.from_pretrained(processor_name)
        self.use_clip_tokens = use_clip_tokens

    def __repr__(self):
        return "CLIPModel({})".format(self.get_config_dict())

    def forward(self, features):
        image_embeds = []
        text_embeds = []

        if 'pixel_values' in features:
            vision_outputs = self.model.vision_model(pixel_values=features['pixel_values'], return_dict=False)
            last_hidden_state, pooled_output, *_ = vision_outputs
            image_embeds = self.model.visual_projection(pooled_output)
            if self.use_clip_tokens:
                image_token_embeds = self.model.visual_projection(last_hidden_state)
                
        if 'input_ids' in features:
            text_outputs = self.model.text_model(
                input_ids=features.get('input_ids'),
                attention_mask=features.get('attention_mask', None),
                position_ids=features.get('position_ids', None),
                output_attentions=features.get('output_attentions', None),
                output_hidden_states=features.get('output_hidden_states', None),
                return_dict=False,
            )
            last_hidden_state, pooled_output, *_ = text_outputs
            text_embeds = self.model.text_projection(pooled_output)
            if self.use_clip_tokens:
                text_token_embeds = self.model.text_projection(last_hidden_state)

        sentence_embedding = []
        image_features = iter(image_embeds)
        text_features = iter(text_embeds)

        for idx, input_type in enumerate(features['image_text_info']):
            if input_type == 0:
                sentence_embedding.append(next(image_features))
            else:
                sentence_embedding.append(next(text_features))

        features['sentence_embedding'] = torch.stack(sentence_embedding).float()

        if self.use_clip_tokens:
            prev_input_type = None
            for input_type in features['image_text_info']:
                if prev_input_type is None:
                    prev_input_type = input_type
                else:
                    assert prev_input_type == input_type
            
            # (batch_size, num_tokens, D)
            if prev_input_type == 0:
                features['token_embeddings'] = image_token_embeds
                features['attention_mask'] = torch.ones(*image_token_embeds.shape[:2])
            else:
                features['token_embeddings'] = text_token_embeds

        return features

    def tokenize(self, texts):
        images = []
        texts_values = []
        image_text_info = []

        for idx, data in enumerate(texts):
            if isinstance(data, Image.Image):  # An Image
                images.append(data)
                image_text_info.append(0)
            else:  # A text
                texts_values.append(data)
                image_text_info.append(1)

        if len(texts_values) == 0:
            texts_values = None
        if len(images) == 0:
            images = None

        inputs = self.processor(text=texts_values, images=images, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs['image_text_info'] = image_text_info
        return inputs
    
    def get_word_embedding_dimension(self) -> int:
        return self.model.text_embed_dim
    
    def get_sentence_embedding_dimension(self) -> int:
        return self.model.projection_dim
    
    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)

        with open(os.path.join(output_path, 'clip_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        config = {}
        config_path = os.path.join(input_path, 'clip_config.json')
        if os.path.exists(config_path):
            config = json.load(open(config_path))

        return CLIPModel(model_name=input_path, **config)
