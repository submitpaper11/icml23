import os

import captum.attr
import numpy
import numpy as np
import pandas as pd
import torch.multiprocessing
import torchray.benchmark.datasets
import torchvision.models
import transformers.models.vit.modeling_vit
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import auc
from scipy import ndimage
from imagenet_lables import label_map
from coco_labels import coco_label_list
from saliency_utils import *
from salieny_models import *
from torchvision.datasets import VOCDetection
from torchvision.datasets import CocoDetection
from evaluation_metrics import evaluations
from vit_model import ViTmodel, ViT_LRP, ViT_explanation_generator

IMAGE_SIZE = 'image_size'

BBOX = 'bbox'

ROOT_IMAGES = "{0}/data/ILSVRC2012_img_val"
IS_VOC = False
IS_COCO = False
IS_VOC_BBOX = False
IS_COCO_BBOX = False
IS_VITMODEL = True
IS_TATAR = False
# IS_HILA = False
INPUT_SCORE = 'score_original_image'
IMAGE_PIXELS_COUNT = 50176
INTERPOLATION_STEPS = 50
# INTERPOLATION_STEPS = 3
ITERATION_STEPS = 5
LABEL = 'label'
TOP_K_PERCENTAGE = 0.25
USE_TOP_K = False
BY_MAX_CLASS = False
GRADUAL_PERTURBATION = True
voc_classes = torchray.benchmark.datasets.VOC_CLASSES
coco_classes = torchray.benchmark.datasets.COCO_CLASSES
IMAGE_PATH = 'image_path'
DATA_VAL_TXT = 'data/val.txt'
# DATA_VAL_TXT = f'data/pics.txt'
device = 'cuda:1'
device = 'cuda'


# device = 'cpu'


def get_images(image_path, interpolation_on_images_steps, is_transformer=False):
    CWD = os.getcwd()
    root = ROOT_IMAGES.format(CWD)

    print(image_path)

    img = Image.open(root + '/' + image_path).convert('RGB')
    im = preprocess(img, is_transformer=is_transformer)
    X = torch.stack([im])

    if interpolation_on_images_steps > 0:
        X = get_interpolated_values(torch.zeros_like(im), im, num_steps=interpolation_on_images_steps)

    return X


def avg_heads_iig(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    # cam = cam.clamp(min=0).mean(dim=0)
    cam = cam.clamp(min=0)
    return cam


def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


def transformer_attribution(vitmodel, feature_extractor, device, label, image, start_layer):
    lrp = ViT_explanation_generator.LRP(vitmodel)
    res = lrp.generate_LRP(image.unsqueeze(0), start_layer=1, method="grad", index=label).reshape(1, 1, 14, 14)
    return res.squeeze().detach()


def generate_map_tattr(model, input, index=None, start_layer=1):
    output = model(input, register_hook=True)
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    # one_hot_vector = one_hot # I changed it.
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    cams = []
    for blk in model.blocks:
        grad = blk.attn.get_attn_gradients()
        cam = blk.attn.get_attention_map()
        with torch.no_grad():
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0).detach())
    rollout = compute_rollout_attention(cams, start_layer=start_layer)
    cam = rollout[:, 0, 1:]
    return cam


# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def generate_map_gae(model, input, index=None):
    output = model(input, register_hook=True)
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    # one_hot_vector = one_hot # I changed it.
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    num_tokens = model.blocks[0].attn.get_attention_map().shape[-1]
    R = torch.eye(num_tokens, num_tokens).cuda()
    for blk in model.blocks:
        grad = blk.attn.get_attn_gradients()
        cam = blk.attn.get_attention_map()
        cam = avg_heads(cam, grad)
        R += apply_self_attention_rules(R.cuda(), cam.cuda()).detach()
    return R[0, 1:]


def get_integrated_grads_vit(vit_ours_model, feature_extractor, device, index, image):
    imgs = get_interpolated_values(torch.zeros_like(image.squeeze().cpu()), image.squeeze().cpu(),
                                   num_steps=ITERATION_STEPS)
    attention_probs = []
    attention_grads = []
    for image in imgs:
        output = vit_ours_model(image.unsqueeze(0).cuda(), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        vit_ours_model.zero_grad()
        one_hot.backward(retain_graph=True)
        cams = []
        for blk in vit_ours_model.blocks:
            grad = blk.attn.get_attn_gradients()
            grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
            grad = grad.clamp(min=0).mean(dim=0)
            cams.append(grad)
        attention_probs.append(torch.stack(cams))
    integrated_attention = torch.stack(attention_probs).squeeze()
    final_attn = torch.mean(integrated_attention, dim=0)

    final_rollout = rollout(
        attentions=final_attn.unsqueeze(1),
        head_fusion="mean",
        gradients=None
    )
    return final_rollout


def get_integrated_attention_vit(vit_ours_model, feature_extractor, device, index, image):
    AGGREGATE_BEFORE = True
    attention_probs = []
    attention_grads = []
    output = vit_ours_model(image.unsqueeze(0).cuda(), register_hook=True)
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * output)
    vit_ours_model.zero_grad()
    one_hot.backward(retain_graph=True)
    cams = []
    for blk in vit_ours_model.blocks:
        grad = blk.attn.get_attn_gradients()
        cam = blk.attn.get_attention_map()
        # cam = avg_heads(cam, grad)[0, 1:]
        cam = avg_heads_iig(cam, grad)
        cams.append(cam)
    attention_probs.append(torch.stack(cams))
    integrated_attention = torch.stack(attention_probs)
    iiattn = get_interpolated_values(torch.zeros_like(integrated_attention.squeeze().cpu()),
                                     integrated_attention.squeeze().cpu(),
                                     num_steps=ITERATION_STEPS)
    cams = []
    grads = []
    for iatn in iiattn:
        iatn.requires_grad = True
        output = vit_ours_model(image.unsqueeze(0).cuda(), register_hook=True, attn_prob=iatn[-1].cuda().unsqueeze(0))
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        vit_ours_model.zero_grad()
        one_hot.backward(retain_graph=True)
        # cams = []
        with torch.no_grad():
            block_cams = []
            block_grads = []
            for blk in vit_ours_model.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attention_map()
                if AGGREGATE_BEFORE:
                    cam = avg_heads_iig(cam, grad)
                    # cam = avg_heads(cam, grad)[0, 1:]
                else:
                    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
                    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
                block_cams.append(cam)
                block_grads.append(grad)
        cams.append(torch.stack(block_cams))
        grads.append(torch.stack(block_grads))
    integrated_attention = torch.stack(cams)
    integrated_gradients = torch.stack(grads)
    final_attn = torch.mean(integrated_attention, dim=0)
    final_rollout = rollout(
        attentions=final_attn.unsqueeze(1),
        head_fusion="mean",
        gradients=None
    )
    return final_rollout


def get_iig_vit(vit_ours_model, feature_extractor, device, index, image):
    AGGREGATE_BEFORE = True
    imgs = get_interpolated_values(torch.zeros_like(image.squeeze().cpu()), image.squeeze().cpu(),
                                   num_steps=ITERATION_STEPS)
    attention_probs = []
    attention_grads = []
    for image in imgs:
        output = vit_ours_model(image.unsqueeze(0).cuda(), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        vit_ours_model.zero_grad()
        one_hot.backward(retain_graph=True)
        cams = []
        block_attn = []
        for blk in vit_ours_model.blocks:
            grad = blk.attn.get_attn_gradients()
            cam = blk.attn.get_attention_map()
            # print(cam.shape)
            block_attn.append(cam)
            # cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
            # grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
            # cam = avg_heads(cam, grad)[0, 1:]
            cam = avg_heads_iig(cam, grad)
            cams.append(cam)

        attn = torch.stack(block_attn)
        iiattn = get_interpolated_values(torch.zeros_like(attn.cpu()), attn.cpu(),
                                         num_steps=ITERATION_STEPS)
        cams = []
        grads = []
        for iatn in iiattn:
            iatn.requires_grad = True
            output = vit_ours_model(image.unsqueeze(0).cuda(), register_hook=True, attn_prob=iatn[-1].cuda())
            if index == None:
                index = np.argmax(output.cpu().data.numpy(), axis=-1)

            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0, index] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)
            vit_ours_model.zero_grad()
            one_hot.backward(retain_graph=True)
            # cams = []
            with torch.no_grad():
                block_cams = []
                block_grads = []
                for blk in vit_ours_model.blocks:
                    grad = blk.attn.get_attn_gradients()
                    cam = blk.attn.get_attention_map()
                    if AGGREGATE_BEFORE:
                        cam = avg_heads_iig(cam, grad)
                        # cam = avg_heads(cam, grad)[0, 1:]
                    else:
                        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
                        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
                    block_cams.append(cam)
                    block_grads.append(grad)
            cams.append(torch.stack(block_cams))
            grads.append(torch.stack(block_grads))
        attention_probs.append(torch.stack(cams))
        attention_grads.append(torch.stack(grads))
    integrated_attention = torch.stack(attention_probs)
    integrated_gradients = torch.stack(attention_grads)
    final_attn = torch.mean(integrated_attention, dim=[0, 1])
    final_grads = torch.mean(integrated_gradients, dim=[0, 1])
    if AGGREGATE_BEFORE:
        final_rollout = rollout(
            attentions=final_attn.unsqueeze(1),
            head_fusion="mean",
            gradients=None
        )
    else:
        final_rollout = rollout(
            attentions=final_attn.unsqueeze(1),
            head_fusion="mean",
            gradients=final_grads.unsqueeze(1)
        )
    return final_rollout


def rollout(
        attentions,
        discard_ratio: float = 0,
        start_layer=0,
        head_fusion="max",
        gradients=None,
        return_resized: bool = True,
):
    all_layer_attentions = []
    attn = []
    if gradients is not None:
        for attn_score, grad in zip(attentions, gradients):
            score_grad = attn_score * grad
            attn.append(score_grad.clamp(min=0).detach())
    else:
        attn = attentions
    for attn_heads in attn:
        if head_fusion == "mean":
            fused_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
        elif head_fusion == "max":
            fused_heads = (attn_heads.max(dim=1)[0]).detach()
        elif head_fusion == "sum":
            fused_heads = (attn_heads.sum(dim=1)[0]).detach()
        elif head_fusion == "median":
            fused_heads = (attn_heads.median(dim=1)[0]).detach()
        elif head_fusion == "min":
            fused_heads = (attn_heads.min(dim=1)[0]).detach()
        flat = fused_heads.view(fused_heads.size(0), -1)
        _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
        # indices = indices[indices != 0]
        flat[0, indices] = 0
        all_layer_attentions.append(fused_heads)
        # print(f'all_layer_attentions_len: {len(all_layer_attentions)}')
        # print(f'all_layer_attentions[0].shape: {all_layer_attentions[0].shape}')
    rollout_arr = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
    mask = rollout_arr[:, 0, 1:]  # result.shape: [1, n_tokens + 1, n_tokens + 1]
    # print(f'mask.shape: {mask.shape}')
    if return_resized:
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width)
    return mask


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = (
        torch.eye(num_tokens)
        .expand(batch_size, num_tokens, num_tokens)
        .to(all_layer_matrices[0].device)
    )
    # print(eye.shape)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [
        all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
        for i in range(len(all_layer_matrices))
    ]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer + 1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention


def top_k_heatmap(heatmap, percent):
    top = int(IMAGE_PIXELS_COUNT * percent)
    heatmap = heatmap.reshape(IMAGE_PIXELS_COUNT)
    ind = np.argpartition(heatmap, -top)[-top:]
    all = np.arange(IMAGE_PIXELS_COUNT)
    rem = [item for item in all if item not in ind]
    heatmap[rem] = 0.
    heatmap = heatmap.reshape(224, 224)
    return heatmap


def image_show(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def calc_score_original(input):
    global label
    if IS_VITMODEL:
        preds_original_image = model(input.to(device)).detach()
    else:
        preds_original_image = model(input.to(device)).logits.detach()

    one_hot = torch.zeros(preds_original_image.shape).to(device)
    one_hot[:, label] = 1

    score_original_image = torch.sum(one_hot * preds_original_image, dim=1).detach()
    return score_original_image


def calc_img_score(img):
    global label
    if not IS_VITMODEL:
        preds_masked_image = model(img.unsqueeze(0).to(device)).logits.detach()
    else:
        preds_masked_image = model(img.unsqueeze(0).to(device)).detach()
    one_hot = torch.zeros(preds_masked_image.shape).to(device)
    one_hot[:, label] = 1
    score = torch.sum(one_hot * preds_masked_image, dim=1).detach()
    return score


def calc_blended_image_score(heatmap):
    global label
    img_cv = tensor2cv(input.squeeze())
    heatmap_cv = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    masked_image = np.uint8((np.repeat(heatmap_cv.reshape(224, 224, 1), 3, axis=2) * img_cv))
    img = preprocess(Image.fromarray(masked_image))
    preds_masked_image = model(img.unsqueeze(0).to(device)).detach()
    one_hot = torch.zeros(preds_masked_image.shape).to(device)
    one_hot[:, label] = 1
    score = torch.sum(one_hot * preds_masked_image, dim=1).detach()
    return score


def coco_calculate_localization():
    global ROOT_IMAGES
    ROOT_IMAGES = "{0}/data/COCO/val2014/"

    # CWD = os.getcwd()
    # ROOT_IMAGES = "{0}/data/COCO/val2014/".format(CWD)
    # ANNOTATION_FILE = "{0}/data/COCO/annotations/instances_val2014.json".format(CWD)
    # coco = CocoDetection(ROOT_IMAGES, ANNOTATION_FILE)
    # coco_classes = torchray.benchmark.datasets.COCO_CLASSES
    # BASE_NAME = 'COCO_val2014_000000000000'
    # JPG = '.jpg'

    # for i, (img, annotation) in tqdm(enumerate(coco)):
    #     print(annotation)
    #     class_id = int(annotation[0]['category_id'])
    #     class_id = max(0, class_id - 1)
    #     image_id = str(annotation[0]['image_id'])
    #     bbox = annotation[0]['bbox']
    #     filename = f'{BASE_NAME[:len(BASE_NAME) - len(image_id)]}{image_id}{JPG}'

    # with open('data/coco_bbox.txt', 'w') as f:
    #     for i, (img, annotation) in tqdm(enumerate(coco)):
    #         # print(annotation)
    #         if len(annotation) > 0:
    #             class_id = int(annotation[0]['category_id'])
    #             class_id = max(0, class_id - 1)
    #             image_id = str(annotation[0]['image_id'])
    #             bbox = annotation[0]['bbox']
    #             filename = f'{BASE_NAME[:len(BASE_NAME) - len(image_id)]}{image_id}{JPG}'
    #             width, height = img.size[0], img.size[1]
    #             if len(bbox) == 4:
    #                 f.write(f'{filename}|{class_id}|{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}|{width},{height}\n')

    images_by_label = {}
    with open(f'data/coco_bbox.txt') as f:
        lines = f.readlines()
        for line in lines:
            file_name, label, bbox_str, size = line.split('|')
            label = int(label)
            bbox_arr = bbox_str.split(',')
            bbox = [float(bbox_arr[0]), float(bbox_arr[1]), float(bbox_arr[2]), float(bbox_arr[3])]
            if label not in images_by_label:
                images_by_label[label] = [{IMAGE_PATH: file_name, BBOX: bbox, IMAGE_SIZE: size}]
            else:
                images_by_label[label].append({IMAGE_PATH: file_name, BBOX: bbox, IMAGE_SIZE: size})
    set_input = []
    for i, (k, v) in tqdm(enumerate(images_by_label.items())):
        if i % 20 == 0:
            label = k
            label_images_paths = v
            for j, image_map in enumerate(label_images_paths):
                if j % 200 == 0:
                    set_input.append({IMAGE_PATH: image_map[IMAGE_PATH], LABEL: label, BBOX: image_map[BBOX],
                                      IMAGE_SIZE: image_map[IMAGE_SIZE]})
    df = pd.DataFrame(set_input)
    return df


def create_set_from_coco():
    # global ROOT_IMAGES
    # CWD = os.getcwd()
    # ROOT_IMAGES = "{0}/data/COCO/val2014/".format(CWD)
    ROOT_IMAGES = "{0}/data/COCO/val2014/"
    # ANNOTATION_FILE = "{0}/data/COCO/annotations/instances_val2014.json".format(CWD)
    # coco = CocoDetection(ROOT_IMAGES, ANNOTATION_FILE)
    # coco_classes = torchray.benchmark.datasets.COCO_CLASSES
    # BASE_NAME = 'COCO_val2014_000000000000'
    # JPG = '.jpg'

    # for i, (img, annotation) in tqdm(enumerate(coco)):
    #     print(annotation)
    #     class_id = int(annotation[0]['category_id'])
    #     class_id = max(0, class_id - 1)
    #     image_id = str(annotation[0]['image_id'])
    #     filename = f'{BASE_NAME[:len(BASE_NAME) - len(image_id)]}{image_id}{JPG}'

    # with open('data/coco.txt', 'w') as f:
    # for i, (img, annotation) in tqdm(enumerate(coco)):
    #     if len(annotation) != 0:
    #         class_id = int(annotation[0]['category_id'])
    #         class_id = max(0, class_id - 1)
    #         image_id = str(annotation[0]['image_id'])
    #         filename = f'{BASE_NAME[:len(BASE_NAME) - len(image_id)]}{image_id}{JPG}'
    #         f.write(f'{filename} {class_id}\n')

    images_by_label = {}
    with open(f'data/coco.txt') as f:
        lines = f.readlines()
        for line in lines:
            file_name, label = line.split()
            label = int(label)
            if label not in images_by_label:
                images_by_label[label] = [file_name]
            else:
                images_by_label[label].append(file_name)
    set_input = []
    for i, (k, v) in tqdm(enumerate(images_by_label.items())):
        if i % 20 == 0:
            label = k
            image_paths = v
            for j, image_path in enumerate(image_paths):
                if j % 500 == 0:
                    set_input.append({IMAGE_PATH: image_path, LABEL: label})
    df = pd.DataFrame(set_input)
    return df


def voc_calculate_localization():
    global ROOT_IMAGES
    ROOT_IMAGES = "{0}/data/VOC/VOCdevkit/VOC2007/JPEGImages"

    CWD = os.getcwd()
    ROOT_IMAGES = "{0}/data/VOC/VOCdevkit/VOC2007/JPEGImages".format(CWD)
    voc = VOCDetection('data/VOC', year='2007', image_set='test')
    voc_classes = torchray.benchmark.datasets.VOC_CLASSES
    # for i, (k, annotation) in tqdm(enumerate(voc)):
    #     print(annotation)
    #     break

    # with open('data/voc_bbox.txt', 'w') as f:
    #     for i, (k, annotation) in tqdm(enumerate(voc)):
    #         filename = annotation['annotation']['filename']
    #         size = annotation['annotation']['size']
    #         width, height = size['width'], size['height']
    #         class_string = annotation['annotation']['object'][0]['name']
    #         class_id = voc_classes.index(class_string)
    #         bbox = annotation['annotation']['object'][0]['bndbox']
    #         bbox = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
    #         f.write(f'{filename}|{class_id}|{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}|{width},{height}\n')

    images_by_label = {}
    with open(f'data/voc_bbox.txt') as f:
        lines = f.readlines()
        for line in lines:
            file_name, label, bbox_str, size = line.split('|')
            label = int(label)
            bbox_arr = bbox_str.split(',')
            bbox = [float(bbox_arr[0]), float(bbox_arr[1]), float(bbox_arr[2]), float(bbox_arr[3])]
            if label not in images_by_label:
                images_by_label[label] = [{IMAGE_PATH: file_name, BBOX: bbox, IMAGE_SIZE: size}]
            else:
                images_by_label[label].append({IMAGE_PATH: file_name, BBOX: bbox, IMAGE_SIZE: size})
    set_input = []
    for i, (k, v) in tqdm(enumerate(images_by_label.items())):
        if i % 20 == 0:
            label = k
            label_images_paths = v
            for j, image_map in enumerate(label_images_paths):
                if j % 200 == 0:
                    set_input.append({IMAGE_PATH: image_map[IMAGE_PATH], LABEL: label, BBOX: image_map[BBOX],
                                      IMAGE_SIZE: image_map[IMAGE_SIZE]})
    df = pd.DataFrame(set_input)
    return df


def create_set_from_voc():
    global ROOT_IMAGES
    CWD = os.getcwd()
    ROOT_IMAGES = "{0}/data/VOC/VOCdevkit/VOC2007/JPEGImages".format(CWD)
    # voc = VOCDetection('data/VOC', year='2007', image_set='test')
    # voc_classes = torchray.benchmark.datasets.VOC_CLASSES

    # with open('data/voc.txt', 'w') as f:
    #     for i, (k, annotation) in tqdm(enumerate(voc)):
    #         filename = annotation['annotation']['filename']
    #         class_string = annotation['annotation']['object'][0]['name']
    #         class_id = voc_classes.index(class_string)
    #         f.write(f'{filename} {class_id}\n')

    images_by_label = {}
    with open(f'data/voc.txt') as f:
        lines = f.readlines()
        for line in lines:
            file_name, label = line.split()
            label = int(label)
            if label not in images_by_label:
                images_by_label[label] = [file_name]
            else:
                images_by_label[label].append(file_name)
    set_input = []
    for i, (k, v) in tqdm(enumerate(images_by_label.items())):
        if i % 6 == 0:
            label = k
            image_paths = v
            for j, image_path in enumerate(image_paths):
                if j % 200 == 0:
                    set_input.append({IMAGE_PATH: image_path, LABEL: label})
    df = pd.DataFrame(set_input)

    return df


def create_set_from_txt():
    images_by_label = {}
    with open(f'data/pics.txt') as f:
        # with open(f'data/two_classes.txt') as f:
        lines = f.readlines()
        for line in lines:
            file_name, label = line.split()
            label = int(label)
            if label not in images_by_label:
                images_by_label[label] = [file_name]
            else:
                images_by_label[label].append(file_name)
    set_input = []
    for i, (k, v) in tqdm(enumerate(images_by_label.items())):
        label = k
        image_paths = v
        for j, image_path in enumerate(image_paths):
            set_input.append({IMAGE_PATH: image_path, LABEL: label})
    df = pd.DataFrame(set_input)

    return df


def create_set_from_txt_prod():
    images_by_label = {}
    with open(DATA_VAL_TXT) as f:
        lines = f.readlines()
        for line in lines:
            file_name, label = line.split()
            label = int(label)
            if label not in images_by_label:
                images_by_label[label] = [file_name]
            else:
                images_by_label[label].append(file_name)
    set_input = []
    for i, (k, v) in tqdm(enumerate(images_by_label.items())):
        label = k
        image_paths = v
        for j, image_path in enumerate(image_paths):
            if j % 5 == 0:
                set_input.append({IMAGE_PATH: image_path, LABEL: label})
    df = pd.DataFrame(set_input)

    return df


def create_set_from_txt_prod_img(aa, bb):
    images_by_label = {}
    with open(DATA_VAL_TXT) as f:
        lines = f.readlines()
        for line in lines:
            file_name, label = line.split()
            label = int(label)
            if label not in images_by_label:
                images_by_label[label] = [file_name]
            else:
                images_by_label[label].append(file_name)
    set_input = []
    for i, (k, v) in tqdm(enumerate(images_by_label.items())):
        label = k
        image_paths = v
        if k > aa and k < bb:
            for j, image_path in enumerate(image_paths):
                if j % 10 == 0:
                    set_input.append({IMAGE_PATH: image_path, LABEL: label})
    df = pd.DataFrame(set_input)

    return df


def plot_image(image) -> None:  # [1,3,224,224] or [3,224,224]
    image = image if len(image.shape) == 3 else image.squeeze(0)
    plt.imshow(image.cpu().detach().permute(1, 2, 0))
    plt.show()


def handle_top10_results(heatmap):
    TEN_PERCENT = 0.1
    top10heatmap = top_k_heatmap(heatmap, TEN_PERCENT)
    top10score = calc_blended_image_score(top10heatmap)
    method_ADP_TOP10 = "TOP_10_ADP|{0}".format(operation)
    current_image_results[method_ADP_TOP10] = max(0, score_original_image.item() - top10score.item())
    method_pic_TOP10 = "TOP_10_PIC|{0}".format(operation)
    if score_original_image.item() < top10score.item():
        current_image_results[method_pic_TOP10] = 1
    else:
        current_image_results[method_pic_TOP10] = 0


def write_imgs_iterate(img_name):
    num_rows = 3
    num_col = 4
    f = plt.figure(figsize=(30, 20))
    plt.subplot(num_rows, num_col, 1)
    plt.imshow(t)
    plt.title('ground truth')
    plt.axis('off')

    i = 2
    for item in img_dict:
        plt.subplot(num_rows, num_col, i)
        plt.imshow(item["image"])
        plt.title(item["title"])
        plt.axis('off')
        i += 1

    # plt.tight_layout()
    if img_name is not None:
        plt.savefig(img_name)

    plt.clf()
    plt.close('all')


class ReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(input, inplace=False)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def save_results_to_csv():
    global df
    df = pd.DataFrame(results)
    df.loc['total'] = df.mean()
    df.loc['fields'] = df.keys()
    df.to_csv(f'{ITERATION}.csv')


def save_results_to_csv_step(step):
    global df
    df = pd.DataFrame(results)
    df.loc['total'] = df.mean()
    df.loc['fields'] = df.keys()
    df.to_csv(f'./csvs/{ITERATION}-{step}.csv')


ITERATION = 'iig-vit'


def write_heatmap(model_name, image_path, operation, heatmap_cv):
    CWD = os.getcwd()
    np.save("{0}/data/heatmaps/{1}_{2}_{3}".format(CWD, image_path[:-5], operation, model_name), heatmap_cv)


def write_mask(model_name, image_path, operation, masked_image):
    CWD = os.getcwd()
    np.save("{0}/data/masks/{1}_{2}_{3}".format(CWD, image_path[:-5], operation, model_name), masked_image)


def handle_transformers(model, operations, image, label, gt_image):
    img_dict = []

    for operation in operations:
        if operation == 'gae':
            gae = generate_map_gae(model, image.unsqueeze(0).to(device), label).reshape(14, 14).detach()
            im, score, heatmap_cv, blended_img_mask, heatmap, t = blend_transformer_heatmap(image, gae)
            evaluations.run_all_evaluations(input, operation, predicted_label, target_label, save_image=save_img,
                                            heatmap=heatmap,
                                            blended_img_mask=blended_img_mask, blended_im=im, t=t, model=model,
                                            result_dict=current_image_results)
            img_dict.append({"image": im, "title": 'gae'})
        elif operation == 't-attr':
            model_tattr = ViT_LRP.vit_base_patch16_224(pretrained=True).to(device)
            # model_tattr = ViT_LRP.vit_small_patch16_224(pretrained=True).to(device)

            t_attr = transformer_attribution(model_tattr, [], device, label, image.to(device), 0).detach()
            im2, score, heatmap_cv, blended_img_mask, heatmap, t = blend_transformer_heatmap(image, t_attr)
            evaluations.run_all_evaluations(input, operation, predicted_label, target_label, save_image=save_img,
                                            heatmap=heatmap,
                                            blended_img_mask=blended_img_mask, blended_im=im2, t=t, model=model,
                                            result_dict=current_image_results)
            img_dict.append({"image": im2, "title": 't_attr'})
        elif operation == 'iig':
            iig_attribution = get_iig_vit(model, [], device, label,
                                          image)
            im3, score, heatmap_cv, blended_img_mask, heatmap, t = blend_transformer_heatmap(image, iig_attribution)
            evaluations.run_all_evaluations(input, operation, predicted_label, target_label, save_image=save_img,
                                            heatmap=heatmap,
                                            blended_img_mask=blended_img_mask, blended_im=im3, t=t, model=model,
                                            result_dict=current_image_results)
            img_dict.append({"image": im3, "title": 'iig'})

        elif operation == 'ablation-attention':
            iig_attribution = get_integrated_attention_vit(model, [], device, label,
                                                           image)
            im3, score, heatmap_cv, blended_img_mask, heatmap, t = blend_transformer_heatmap(image, iig_attribution)
            evaluations.run_all_evaluations(input, operation, predicted_label, target_label, save_image=save_img,
                                            heatmap=heatmap,
                                            blended_img_mask=blended_img_mask, blended_im=im3, t=t, model=model,
                                            result_dict=current_image_results)
            img_dict.append({"image": im3, "title": 'ablation-attention'})
        elif operation == 'ablation-image':
            iig_attribution = get_integrated_grads_vit(model, [], device, label,
                                                       image)
            im3, score, heatmap_cv, blended_img_mask, heatmap, t = blend_transformer_heatmap(image, iig_attribution)
            evaluations.run_all_evaluations(input, operation, predicted_label, target_label, save_image=save_img,
                                            heatmap=heatmap,
                                            blended_img_mask=blended_img_mask, blended_im=im3, t=t, model=model,
                                            result_dict=current_image_results)
            img_dict.append({"image": im3, "title": 'ablation-image'})
    t = tensor2cv(gt_image.squeeze())
    string_lbl = label_map.get(int(label))
    name = f'qualitive_results/{model_name}_{string_lbl}_{image_path}'
    save_transformer_images(img_dict, t, name)


def save_transformer_images(img_dict, t, img_name):
    num_rows = 3
    num_col = 4
    f = plt.figure(figsize=(30, 20))
    plt.subplot(num_rows, num_col, 1)
    plt.imshow(t)
    plt.title('ground truth')
    plt.axis('off')

    i = 2
    for item in img_dict:
        plt.subplot(num_rows, num_col, i)
        plt.imshow(item["image"])
        plt.title(item["title"])
        plt.axis('off')
        i += 1

    # plt.tight_layout()
    if img_name is not None:
        plt.savefig(img_name)


def blend_transformer_heatmap(image, x1):
    heatmap = x1.unsqueeze(0).unsqueeze(0)
    # heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
    heatmap = torch.nn.functional.interpolate(heatmap, scale_factor=16, mode='bilinear')
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().cpu().data.numpy()
    t = tensor2cv(image)
    im, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap, use_mask=True)

    # blended_img_mask = blended_img_mask.transpose((2,0,1))
    return im, score, heatmap_cv, blended_img_mask, heatmap, t


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    results = []
    image = None
    model_name = 'vit-base'
    # model_name = 'vit-small'
    methods = ['iig', 'gae']
    methods = ['iig', 't-attr', 'gae']
    num_layers_options = [1]
    USE_MASK = True
    torch.nn.modules.activation.ReLU.forward = ReLU.forward

    # model = GradModel(model_name, feature_layer=FEATURE_LAYER_NUMBER)
    # model.to(device)
    # model.eval()
    # model.zero_grad()

    if model_name == 'vit-base':
        model = ViTmodel.vit_base_patch16_224(pretrained=True).to(device)
    else:
        model = ViTmodel.vit_small_patch16_224(pretrained=True).to(device)

    df = create_set_from_txt()

    print(len(df))
    df_len = len(df)
    for index, row in tqdm(df.iterrows()):
        current_image_results = {}
        image_path = row[IMAGE_PATH]
        label = row[LABEL]
        target_label = label
        input_transformers = get_images(image_path, 0, is_transformer=True)
        gt_image = get_images(image_path, 0)
        input = input_transformers
        input_predictions = model(input.to(device))
        if IS_VITMODEL:
            predicted_label = torch.max(input_predictions, 1).indices[0].item()
        else:
            predicted_label = torch.max(input_predictions.logits, 1).indices[0].item()

        save_img = True

        # print(voc_classes[predicted_label])
        if BY_MAX_CLASS:
            label = predicted_label
        # try:
        for num_layers_option in num_layers_options:
            save_img = False
            save_heatmaps_masks = False
            to_write_results = True
            operation_index = 0
            score_original_image = 0
            img_dict = []
            score_original_image = calc_score_original(input)
            handle_transformers(model, methods, input_transformers.squeeze(), label, gt_image)
        if save_img:
            string_lbl = label_map.get(int(label))
            if IS_VOC:
                string_lbl = voc_classes[int(label)]
            if IS_COCO:
                string_lbl = coco_label_list[int(label)]
            write_imgs_iterate(f'qualitive_results/{model_name}_{string_lbl}_{image_path}')

        current_image_results[IMAGE_PATH] = image_path
        current_image_results[LABEL] = label
        current_image_results[INPUT_SCORE] = score_original_image.item()

        results.append(current_image_results)
        torch.cuda.empty_cache()
        if index % 100 == 0:
            if to_write_results:
                save_results_to_csv_step(index)
            print(index)
        # except Exception as e:
        #     print(f'there was an error with image {image_path}...{e}')
    if to_write_results:
        save_results_to_csv()
