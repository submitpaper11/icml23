import os
from glob import glob

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image, ImageFilter
from torchvision.datasets import ImageNet

import methods
from coco_dataset import Coco_Segmentation
from segmentation_utils.metrices import batch_intersection_union
from segmentation_utils.metrices import batch_pix_accuracy
from segmentation_utils.metrices import get_ap_scores


def apply_threshold(map):
    meanval = map.flatten().mean()
    new = np.where(map > meanval, 255, 0).astype(np.uint8)
    return new


def save_results_to_csv(results, fname):
    global df
    df = pd.DataFrame(results)
    df.loc['total'] = df.mean()
    df.loc['fields'] = df.keys()
    df.to_csv(f'{fname}.csv')


def eval_batch(Res, labels):
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    ret = Res.mean()

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1 - Res

    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0

    # TEST
    pred = Res.clamp(min=0) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()
    # print("target", target.shape)

    output = torch.cat((Res_0, Res_1), 1)
    output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)

    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap, batch_f1 = 0, 0

    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    # print("output", output.shape)
    # print("ap labels", labels.shape)
    # ap = np.nan_to_num(get_ap_scores(output, labels))
    ap = np.nan_to_num(get_ap_scores(output_AP, labels))
    # f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
    batch_ap += ap
    # batch_f1 += f1

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target


def eval_batch2(heatmap, labels):
    Res = torch.tensor(heatmap).unsqueeze(0)
    # threshold between FG and BG is the mean
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    ret = Res.mean()

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1 - Res

    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0

    # TEST
    threshold = 0.
    pred = Res.clamp(min=threshold) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()
    # print("target", target.shape)

    output = torch.cat((Res_0, Res_1), 0)
    output_AP = torch.cat((Res_0_AP, Res_1_AP), 0)
    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap, batch_f1 = 0, 0

    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    # print("output", output.shape)
    # print("ap labels", labels.shape)
    # ap = np.nan_to_num(get_ap_scores(output, labels))
    ap = np.nan_to_num(get_ap_scores(output_AP.unsqueeze(0), labels))
    batch_ap += ap
    batch_f1 += 0

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target


def init_get_normalize_and_trns():
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    test_img_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    test_img_trans_only_resize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_lbl_trans = transforms.Compose([
        transforms.Resize((224, 224), Image.NEAREST),
    ])

    return test_img_trans, test_img_trans_only_resize, test_lbl_trans


import matplotlib.pyplot as plt


def image_show(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

ITERATION = 'iig'
if __name__ == '__main__':
    import torchvision.transforms as transforms
    from tqdm import tqdm

    # from imageio import imsave
    # import scipy.io as sio
    device = 'cuda'
    # device = 'cpu'
    # Data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_img_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    test_img_trans, test_img_trans_only_resize, test_lbl_trans = init_get_normalize_and_trns()
    CWD = os.getcwd()
    COCO_SEG_PATH = f'{CWD}/data/COCO/2017'
    # COCO_SEG_PATH = f'/data/COCO/2017'

    ds = Coco_Segmentation(COCO_SEG_PATH,
                           transform=test_img_trans,
                           transform_resize=test_img_trans_only_resize, target_transform=test_lbl_trans)

    test_lbl_trans = transforms.Compose([
        transforms.Resize((224, 224), Image.NEAREST),
    ])


    operations = ['iig', 'lift-cam', 'ablation-cam', 'gradcam', 'gradcampp']
    operations = ['iig', 'gradcam']

    segmentation_results = {}
    for operation in operations:
        segmentation_results[f'{operation}_IoU'] = 0
        segmentation_results[f'{operation}_mAP'] = 0
        segmentation_results[f'{operation}_pixAcc'] = 0
    results = []
    total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    total_ap, total_f1 = [], []

    for i, (img_norm, target, img_resize) in enumerate(tqdm(ds)):
        tgt = target
        img = img_resize
        segmentation_results = {}

        models = ['densnet', 'convnext', 'resnet101', 'resnet18']
        layer_options = [12, 8]
        model_name = models[-1]
        FEATURE_LAYER_NUMBER = layer_options[-1]

        heatmaps = methods.generate_heatmap(models[-1], layer_options[-1], operations, img, device=device)
        op_idx = 0
        for operation in operations:
            map = heatmaps[op_idx]
            # model.zero_grad()
            correct, labeled, inter, union, ap, f1, pred, target = eval_batch(
                torch.tensor(map).unsqueeze(0).unsqueeze(0),
                torch.tensor(tgt).unsqueeze(0))

            total_correct += correct.astype('int64')
            total_label += labeled.astype('int64')
            total_inter += inter.astype('int64')
            total_union += union.astype('int64')
            total_ap += [ap]
            total_f1 += [f1]
            pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
            IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
            mIoU = IoU.mean()
            mAp = np.mean(total_ap)
            mF1 = np.mean(total_f1)
            segmentation_results[f'{operation}_IoU'] = mIoU
            segmentation_results[f'{operation}_mAP'] = mAp
            segmentation_results[f'{operation}_pixAcc'] = pixAcc
            op_idx += 1
        results.append(segmentation_results)
        if i % 300 == 0:
            save_results_to_csv(results, f'segmentation-{ITERATION}-{i}')
    save_results_to_csv(results, f'segmentation-{ITERATION}')
