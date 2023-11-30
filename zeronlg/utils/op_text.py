import torch
from torch import Tensor
from typing import Dict


def random_masking_(
        tokenizer,
        tokenized_results: Dict[str, Tensor],
        mask_prob: float = 0.15,
        replace_prob: float = 0.8,
        random_prob: float = 0.1, 
    ):
    """
    in-place operation to randomlly mask the input ids of tokenized results

    :param tokenized_results: Obtained by tokenzier
    :param mask_prob: Probability to mask a token
    :param replace_prob: Probability to replace masked token with [MASK]
    :param random_prob: Probability to replace masked token with a random token
    """
    if len(tokenized_results) == 0:
        return
    
    input_ids = tokenized_results['input_ids']
    special_tokens_mask = [tokenizer.get_special_tokens_mask(ids.tolist(), already_has_special_tokens=True) for ids in input_ids]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix = torch.full(input_ids.shape, mask_prob)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, replace_prob)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    current_prob = random_prob / (1 - replace_prob)
    indices_random = torch.bernoulli(torch.full(input_ids.shape, current_prob)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), input_ids.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]
