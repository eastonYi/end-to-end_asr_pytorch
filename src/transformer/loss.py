import torch
import torch.nn.functional as F


def cal_ce_loss(logits, targets, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """

    logits = logits.view(-1, logits.size(2))
    targets = targets.contiguous().view(-1)
    if smoothing > 0.0:
        eps = smoothing
        n_class = logits.size(1)

        # Generate one-hot matrix: N x C.
        # Only label position is 1 and all other positions are 0
        # gold include -1 value (0) and this will lead to assert error
        one_hot = torch.zeros_like(logits).scatter(1, targets.long().view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
        log_prb = F.log_softmax(logits, dim=1)

        non_pad_mask = targets.ne(0)
        n_word = non_pad_mask.long().sum()
        loss = -(one_hot * log_prb).sum(dim=1)
        ce_loss = loss.masked_select(non_pad_mask).sum() / n_word
    else:
        ce_loss = F.cross_entropy(logits, targets,
                                  ignore_index=0,
                                  reduction='mean')

    return ce_loss


def cal_ctc_ce_loss(logits_ctc, len_logits_ctc, logits_ce, targets, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    len_logits_ctc, logits_ctc, logits_ce = model_output
    """
    # ctc loss
    n_class = logits_ctc.size(-1)
    target_lengths = targets.ne(0).int().sum(1)
    ctc_log_probs = F.log_softmax(logits_ctc, dim=-1).transpose(0,1)
    ctc_loss = F.ctc_loss(ctc_log_probs, targets,
                          len_logits_ctc, target_lengths, blank=n_class-1)

    # ce loss
    ce_loss = cal_ce_loss(logits_ce, targets, smoothing)

    return ctc_loss, ce_loss


def cal_ctc_qua_ce_loss(logits_ctc, len_logits_ctc, _number, number, logits_ce, targets, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """
    # qua loss
    qua_loss = torch.pow(_number - number, 2).mean()

    # ce loss
    ctc_loss, ce_loss = cal_ctc_ce_loss(
        logits_ctc, len_logits_ctc, logits_ce, targets, smoothing)

    return qua_loss, ctc_loss, ce_loss
