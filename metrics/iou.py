#
import os
import torch
from torchvision import transforms
import cv2
from toolbox.msg import runMsg

if __name__ == "__main__":
    # get img file in a list
    label_path = \
        '/home/RGBDMirrorSegmentation/RGBD-Mirror/test/GT_416'
    pre_path = '/home/wby/Desktop/predict_model'
    img_list = os.listdir(pre_path)
    trans = transforms.Compose([transforms.ToTensor()])
    avg_iou, img_num = 0.0, 0.0
    running_metrics_val = runMsg()
    for i, name in enumerate(img_list):
        pred = cv2.imread(os.path.join(pre_path, name), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(label_path, name), cv2.IMREAD_GRAYSCALE)
        pred = trans(pred)
        gt = trans(gt)
        # pred = (pred >= 0.5)
        # gt = (gt >= 0.5)
        # iou = torch.sum((pred & gt)) / torch.sum((pred | gt))
        # if iou == iou:  # for Nan
        #     avg_iou += iou
        #     img_num += 1.0
        # running_metrics_val.update(gt.float(), pred.float())
        running_metrics_val.update(gt, pred)
    # avg_iou /= img_num
    # print(avg_iou.item())
    metrics = running_metrics_val.get_scores()
    print('overall metrics .....')
    iou = metrics["iou: "].item() * 100
    ber = metrics["ber: "].item() * 100
    mae = metrics["mae: "].item()
    F_measure = metrics["F_measure: "].item()
    print('iou:', iou, 'ber:', ber, 'mae:', mae, 'F_measure:', F_measure)



