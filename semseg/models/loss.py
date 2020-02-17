from torch.nn import CrossEntropyLoss


def fcn_loss(inputs, targets):
    """

    Parameters
    ----------
    inputs
    targets

    Returns
    -------

    """
    cel = CrossEntropyLoss()
    (m, c, h, w) = inputs.shape
    i = inputs.permute(0,2,3,1).view(m*h*w, c)
    t = targets.permute(0,2,3,1).view(m*h*w).long()
    return cel(i,t)
