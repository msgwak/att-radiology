def IoU_loss(outputs, labels, flags, eps=1e-6):
    intersection = (outputs * labels).sum(dim=(1,2,3)).clamp(min=eps)
    union = (outputs + labels).clamp(max=1).sum(dim=(1,2,3)).clamp(min=eps)
    loss = 1 - (intersection / union)
    if flags.sum() == 0:
        loss = 0
    else:
        loss = loss[flags].mean()
    return loss