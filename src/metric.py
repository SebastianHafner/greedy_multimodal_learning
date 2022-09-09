import torch


def true_pos(y_true: torch.Tensor, y_pred: torch.Tensor):
    return torch.sum(y_true * torch.round(y_pred))


def false_pos(y_true, y_pred):
    return torch.sum((1. - y_true) * torch.round(y_pred))


def false_neg(y_true: torch.Tensor, y_pred: torch.Tensor):
    return torch.sum(y_true * (1. - torch.round(y_pred)))


def precision(y_true: torch.Tensor, y_pred: torch.Tensor):
    TP = true_pos(y_true, y_pred)
    FP = false_pos(y_true, y_pred)
    denom = TP + FP
    denom = torch.clamp(denom, 10e-05)
    return TP / denom


def recall(y_true: torch.Tensor, y_pred: torch.Tensor):
    TP = true_pos(y_true, y_pred)
    FN = false_neg(y_true, y_pred)
    denom = TP + FN
    denom = torch.clamp(denom, 10e-05)
    return true_pos(y_true, y_pred) / denom


def f1(gts: torch.Tensor, preds: torch.Tensor):
    gts = gts.float().flatten()
    preds = preds.float().flatten()

    with torch.no_grad():
        recall_val = recall(gts, preds)
        precision_val = precision(gts, preds)
        denom = torch.clamp((recall_val + precision_val), 10e-5)

        f1_score = 2. * recall_val * precision_val / denom

    return f1_score


def iou(gts: torch.Tensor, preds: torch.Tensor):
    gts = gts.float().flatten()
    preds = preds.float().flatten()

    with torch.no_grad():
        TP = true_pos(gts, preds)
        FP = false_pos(gts, preds)
        FN = false_neg(gts, preds)

        iou = TP / torch.clamp(TP + FP + FN, 10e-5)

    return iou
