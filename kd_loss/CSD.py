import torch.nn as nn

def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)

def PCL(SS, TS, eps=1e-8):
    return cosine_similarity(SS - SS.mean(1).unsqueeze(1), TS - TS.mean(1).unsqueeze(1), eps)

def InterCSL(Ys, Yt):
    return 1 - PCL(Ys, Yt).mean()

def IntraCSL(Ys, Yt):
    return InterCSL(Ys.transpose(0, 1), Yt.transpose(0, 1))

class CSD(nn.Module):
    def __init__(self):
        super(CSD, self).__init__()
    def forward(self, Ys, Yt):
        assert Ys.ndim in (2, 4)
        if Ys.ndim == 4:
            num_classes = Ys.shape[1]
            Ys = Ys.transpose(1, 3).reshape(-1, num_classes)
            Yt = Yt.transpose(1, 3).reshape(-1, num_classes)
        Ys = Ys.softmax(dim=1)
        Yt = Yt.softmax(dim=1)
        inter_loss = InterCSL(Ys, Yt)
        intra_loss = IntraCSL(Ys, Yt)
        loss = inter_loss + intra_loss
        return loss