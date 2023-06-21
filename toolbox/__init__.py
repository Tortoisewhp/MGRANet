from .metrics import averageMeter, runningScore
from .log import get_logger

from .optim.AdamW import AdamW
from .optim.Lookahead import Lookahead
from .optim.RAdam import RAdam
from .optim.Ranger import Ranger

from .losses.loss import CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, \
    ProbOhemCrossEntropy2d, FocalLoss2d, LovaszSoftmax, LDAMLoss, MscCrossEntropyLoss

from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, \
    compute_speed, setup_seed, group_weight_decay


def get_dataset(cfg):
    assert cfg['dataset'] in ['mirrorRGBD']

    if cfg['dataset'] == 'mirrorRGBD':
        from .datasets.mirrorrgbd import MirrorRGBD
        return MirrorRGBD(cfg, mode='train'), MirrorRGBD(cfg, mode='test')


def get_model(cfg):
    if cfg['model_name'] == 'MGRANet_student':
        from toolbox.models.MGRANet.MGRANet_student import MGRANet_student
        return MGRANet_student()

    elif cfg['model_name'] == 'MGRANet_teacher':
        from toolbox.models.MGRANet.MGRANet_teacher import MGRANet_teacher
        return MGRANet_teacher()