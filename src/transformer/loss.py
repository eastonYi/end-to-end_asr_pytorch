import torch
import torch.nn.functional as F


def cal_ce_loss(pred, gold, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """
    pred = pred.view(-1, pred.size(2))
    gold = gold.contiguous().view(-1)
    if smoothing > 0.0:
        eps = smoothing
        n_class = pred.size(1)

        # Generate one-hot matrix: N x C.
        # Only label position is 1 and all other positions are 0
        # gold include -1 value (0) and this will lead to assert error
        gold_for_scatter = gold.ne(0).long() * gold
        one_hot = torch.zeros_like(pred).scatter(1, gold_for_scatter.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(0)
        n_word = non_pad_mask.sum().item()
        loss = -(one_hot * log_prb).sum(dim=1)
        ce_loss = loss.masked_select(non_pad_mask).sum() / n_word
    else:
        ce_loss = F.cross_entropy(pred, gold,
                                  ignore_index=0,
                                  reduction='elementwise_mean')

    return ce_loss


def cal_ctc_ce_loss(pred, gold, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """
    ctc_pred_len, ctc_pred, pred = pred

    # ctc loss
    n_class = pred.size(-1)
    target_lengths = gold.ne(0).sum(dim=1).int()
    ctc_log_probs = F.log_softmax(ctc_pred, dim=-1).transpose(0,1)
    ctc_loss = F.ctc_loss(ctc_log_probs, gold,
                          ctc_pred_len, target_lengths, blank=n_class-1)

    # ce loss
    ce_loss = cal_ce_loss(pred, gold)

    return ctc_loss, ce_loss
