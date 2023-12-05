#
import os
import time
from tqdm import tqdm
import json
import torch
from torch.utils.data import DataLoader
from toolbox import get_model
from toolbox import averageMeter
from torchvision import transforms
import cv2
from toolbox.datasets.mirrorrgbd import MirrorRGBD
from toolbox.msg import runMsg
def evaluate(logdir, save_predict=False, options=['train', 'val', 'test', 'test_day', 'test_night', "test_withglass", "test_withoutglass"], prefix=''):
    cfg = None
    for file in os.listdir(logdir):
        if file.endswith('.json'):
            with open(os.path.join(logdir, file), 'r') as fp:
                cfg = json.load(fp)
    assert cfg is not None
    device = torch.device("cuda:0")
    loaders = []
    for opt in options:
        dataset = MirrorRGBD(cfg, mode=opt)
        loaders.append((opt, DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])))
        cmap = dataset.cmap
    model = get_model(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(logdir, prefix+'.pth'), map_location={'cuda:0': 'cuda:0'}))
    to_pil = transforms.ToPILImage()
    # running_metrics_val = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    running_metrics_val = runMsg()
    time_meter = averageMeter()
    save_path = os.path.join(logdir, 'predict_result/')
    if not os.path.exists(save_path) and save_predict:
        os.mkdir(save_path)
    for name, test_loader in loaders:
        running_metrics_val.reset()
        print('#'*50 + '    ' + name+prefix + '    ' + '#'*50)
        with torch.no_grad():
            model.eval()
            for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
                time_start = time.time()
                if cfg['inputs'] == 'rgb':
                    image = sample['image'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image)[0]
                else:
                    image = sample['image'].to(device)
                    depth = sample['depth'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image, depth)[0][-1]
                running_metrics_val.update(label.cpu().float(), predict.cpu().float())
                predict = predict.squeeze()
                cv2.imwrite(os.path.join(save_path, sample['label_path'][0]), predict.cpu().numpy()*255)
                time_meter.update(time.time() - time_start, n=image.size(0))
        metrics = running_metrics_val.get_scores()
        print('overall metrics .....')
        iou = metrics["iou: "].item() * 100
        ber = metrics["ber: "].item() * 100
        mae = metrics["mae: "].item()
        print('iou:', iou, 'ber:', ber, 'mae:', mae)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--logdir", type=str, default="/home/MGRANet/toolbox/models/MGRANet/")
    parser.add_argument("-s", type=bool, default=True, help="save predict or not")
    args = parser.parse_args()
    evaluate(args.logdir, save_predict=args.s, options=['test'], prefix='MGRANet_teacher')
