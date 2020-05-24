import torch
import torch.nn.functional as F

from utils import IGNORE_ID


def cal_performance(pred, gold, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    Args:
        pred: N x T x C, score before softmax
        gold: N x T
    """
    ctc_loss, ce_loss = cal_loss(pred, gold, smoothing)

    pred = pred[-1]
    pred = pred.view(-1, pred.size(2))
    gold = gold.contiguous().view(-1)
    pred = pred.max(1)[1]
    non_pad_mask = gold.ne(IGNORE_ID)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return ctc_loss, ce_loss, n_correct


def cal_loss(pred, gold, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """
    if type(pred) == list:
        ctc_pred_len, ctc_pred, pred = pred
        n_class = pred.size(-1)
        target_lengths = gold.ne(IGNORE_ID).sum(dim=1).int()
        ctc_log_probs = F.log_softmax(ctc_pred, dim=-1).transpose(0,1)
        ctc_loss = F.ctc_loss(ctc_log_probs, gold,
                              ctc_pred_len, target_lengths, blank=n_class-1)

    else:
        ctc_loss = 0.0

    pred = pred.view(-1, pred.size(2))
    gold = gold.contiguous().view(-1)
    if smoothing > 0.0:
        eps = smoothing
        n_class = pred.size(1)

        # Generate one-hot matrix: N x C.
        # Only label position is 1 and all other positions are 0
        # gold include -1 value (IGNORE_ID) and this will lead to assert error
        gold_for_scatter = gold.ne(IGNORE_ID).long() * gold
        one_hot = torch.zeros_like(pred).scatter(1, gold_for_scatter.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(IGNORE_ID)
        n_word = non_pad_mask.sum().item()
        loss = -(one_hot * log_prb).sum(dim=1)
        ce_loss = loss.masked_select(non_pad_mask).sum() / n_word
    else:
        ce_loss = F.cross_entropy(pred, gold,
                                  ignore_index=IGNORE_ID,
                                  reduction='elementwise_mean')

    return ctc_loss, ce_loss
