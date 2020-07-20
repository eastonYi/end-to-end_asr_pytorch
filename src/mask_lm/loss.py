import torch
import torch.nn.functional as F


def cal_ce_mask_loss(logits, targets, mask, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """

    logits = logits.view(-1, logits.size(2))
    targets = targets.contiguous().view(-1)
    mask = mask.contiguous().view(-1)

    eps = smoothing
    n_class = logits.size(1)

    # Generate one-hot matrix: N x C.
    # Only label position is 1 and all other positions are 0
    # gold include -1 value (0) and this will lead to assert error
    one_hot = torch.zeros_like(logits).scatter(1, targets.long().view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
    log_prb = F.log_softmax(logits, dim=1)

    non_pad_mask = targets.ne(0)
    concerned_mask = non_pad_mask * mask
    n_word = concerned_mask.long().sum()
    loss = -(one_hot * log_prb).sum(dim=1)


    ce_loss = loss.masked_select(non_pad_mask).sum() / n_word

    return ce_loss


def cal_ctc_loss(logits, len_logits, gold, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """
    n_class = logits.size(-1)
    target_lengths = gold.ne(0).sum(dim=1).int()
    ctc_log_probs = F.log_softmax(logits, dim=-1).transpose(0,1)
    ctc_loss = F.ctc_loss(ctc_log_probs, gold,
                          len_logits, target_lengths, blank=n_class-1)

    return ctc_loss
