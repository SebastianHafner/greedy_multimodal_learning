import torch


def blend_loss(y_hat, y, labeled):
    loss_func = power_jaccard_loss
    losses = []
    for y_pred in y_hat:
        losses.append(loss_func(y_pred, y))

    return sum(losses)


def mmcr_loss(y_hat, y, labeled, alpha: float = 0.1):
    loss_func = power_jaccard_loss
    losses = []
    if labeled.any():
        for y_pred in y_hat:
            losses.append(loss_func(y_pred[labeled, ], y[labeled, ]))
    if not labeled.all():
        not_labeled = torch.logical_not(labeled)
        losses.append(loss_func(y_hat[1][not_labeled, ], y_hat[0][not_labeled, ]) * alpha)
    return sum(losses)


def power_jaccard_loss(input: torch.Tensor, target: torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat ** 2 + tflat ** 2).sum() - (iflat * tflat).sum() + eps

    return 1 - (intersection / denom)