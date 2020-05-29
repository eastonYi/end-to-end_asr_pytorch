import torch.nn.functional as F

from utils.utils import IGNORE_ID


def cal_performance(pred, gold, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    Args:
        pred: N x T x C, score before softmax
        gold: N x T
    """
    ctc_loss = cal_loss(pred, gold, smoothing)
    
    return ctc_loss


def cal_loss(pred, gold, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """
    pred, pred_len = pred
    n_class = pred.size(-1)
    target_lengths = gold.ne(IGNORE_ID).sum(dim=1).int()
    ctc_log_probs = F.log_softmax(pred, dim=-1).transpose(0,1)
    ctc_loss = F.ctc_loss(ctc_log_probs, gold,
                          pred_len, target_lengths, blank=n_class-1)

    return ctc_loss
