import torch.nn.functional as F


def cal_loss(logits, len_logits, gold, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """
    n_class = logits.size(-1)
    target_lengths = gold.ne(0).sum(dim=1).int()
    ctc_log_probs = F.log_softmax(logits, dim=-1).transpose(0,1)
    ctc_loss = F.ctc_loss(ctc_log_probs, gold,
                          len_logits, target_lengths, blank=n_class-1)

    return ctc_loss
