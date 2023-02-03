import os

import captum.attr
import numpy
import numpy as np
import pandas as pd
import torch.multiprocessing
import torchray.benchmark.datasets
import torchvision.models
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
from torchgc.pytorch_grad_cam.score_cam import ScoreCAM
from torchgc.pytorch_grad_cam.eigen_cam import EigenCAM
from torchgc.pytorch_grad_cam.fullgrad_cam import FullGrad
from torchgc.pytorch_grad_cam.layer_cam import LayerCAM
from torchgc.pytorch_grad_cam.ablation_cam import AblationCAM
from torchgc.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from saliency_lib.saliency import *
from evaluation_metrics import evaluations

IMAGE_SIZE = 'image_size'

BBOX = 'bbox'

ROOT_IMAGES = "{0}/data/ILSVRC2012_img_val"
IS_VOC = False
IS_COCO = False
IS_VOC_BBOX = False
IS_COCO_BBOX = False
INPUT_SCORE = 'score_original_image'
IMAGE_PIXELS_COUNT = 50176
INTERPOLATION_STEPS = 50
# INTERPOLATION_STEPS = 3
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

device = 'cpu'
device = 'cuda'


def get_grads_wrt_image(model, label, images_batch, device='cuda', steps=50):
    model.eval()
    model.zero_grad()

    images_batch.requires_grad = True
    preds = model(images_batch.to(device), hook=True)
    _, predicted = torch.max(preds.data, 1)
    # print(f'True label {label}, predicted {predicted}')
    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    score = torch.sum(one_hot * preds)
    score.backward()
    with torch.no_grad():
        # gradients = model.get_activations_gradient()
        image_grads = images_batch.grad.detach()
    # del score
    images_batch.requires_grad = False
    return image_grads


def backward_class_score_and_get_activation_grads(model, label, x, only_post_features=False, device='cuda'):
    model.zero_grad()
    # print(x.shape)
    preds = model(x.to(device), hook=True, only_post_features=only_post_features)
    _, predicted = torch.max(preds.data, 1)
    # print(f'True label {label}, predicted {predicted}')
    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    # score = torch.sum(one_hot * preds, dim=1)
    score = torch.sum(one_hot * preds)
    # score.backward(torch.ones_like(score))
    score.backward()

    # preds.to(device)
    # one_hot.to(device)
    activations_gradients = model.get_activations_gradient().unsqueeze(
        1).detach().cpu()
    # del score

    return activations_gradients


def backward_class_score_and_get_images_grads(model, label, x, only_post_features=False, device='cuda'):
    model.zero_grad()
    preds = model(x.squeeze(1).to(device), hook=True)
    _, predicted = torch.max(preds.data, 1)
    # print(f'True label {label}, predicted {predicted}')
    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    # score = torch.sum(one_hot * preds, dim=1)
    score = torch.sum(one_hot * preds)
    # score.backward(torch.ones_like(score))
    score.backward()

    # preds.to(device)
    # one_hot.to(device)
    images_gradients = model.get_activations_gradient().unsqueeze(
        1).detach().cpu()
    # del score

    return images_gradients


def get_blurred_values(target, num_steps):
    """this function returns a list of all the images interpolation steps."""
    num_steps += 1
    if num_steps <= 0: return np.array([])
    target = target.squeeze()
    tshape = len(target.shape)
    # print(tshape)
    blurred_images_list = []
    for step in range(num_steps):
        sigma = int(step) / int(num_steps)
        sigma_list = [sigma, sigma, 0]

        if tshape == 4:
            sigma_list = [sigma, sigma, sigma, 0]

        blurred_image = ndimage.gaussian_filter(
            target.detach().cpu().numpy(), sigma=sigma_list, mode="grid-constant")
        blurred_images_list.append(blurred_image)

    return numpy.array(blurred_images_list)


def get_images(image_path, interpolation_on_images_steps):
    CWD = os.getcwd()
    root = ROOT_IMAGES.format(CWD)

    print(image_path)

    img = Image.open(root + '/' + image_path).convert('RGB')
    im = preprocess(img)
    X = torch.stack([im])

    if interpolation_on_images_steps > 0:
        X = get_interpolated_values(torch.zeros_like(im), im, num_steps=interpolation_on_images_steps)

    return X


def get_images_blur(image_path, interpolation_on_images_steps):
    CWD = os.getcwd()
    root = ROOT_IMAGES.format(CWD)

    print(image_path)

    img = Image.open(root + '/' + image_path).convert('RGB')
    im = preprocess(img)
    X = torch.stack([im])

    if interpolation_on_images_steps > 0:
        X = torch.tensor(get_blurred_values(im.detach(),
                                            interpolation_on_activations_steps_arr[-1]))

    return X


def get_by_class_saliency_iig_ablation_study_ac(image_path,
                                                label,
                                                operations,
                                                model_name='densnet',
                                                layers=[12],
                                                interpolation_on_images_steps_arr=[0, 50],
                                                interpolation_on_activations_steps_arr=[0, 50],
                                                device='cuda',
                                                use_mask=False):
    images, integrated_heatmaps = heatmap_of_layer_acts(device, image_path, interpolation_on_activations_steps_arr,
                                                        interpolation_on_images_steps_arr, label, layers, model_name)

    heatmap = make_resize_norm(integrated_heatmaps)

    last_image = images[-1]
    t = tensor2cv(last_image)
    im, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap, use_mask=use_mask)

    return t, im, heatmap_cv, blended_img_mask, last_image, score, heatmap


def get_by_class_saliency_iig_ablation_study_im(image_path,
                                                label,
                                                operations,
                                                model_name='densnet',
                                                layers=[12],
                                                interpolation_on_images_steps_arr=[0, 50],
                                                interpolation_on_activations_steps_arr=[0, 50],
                                                device='cuda',
                                                use_mask=False):
    images, integrated_heatmaps = heatmap_of_layer_images(device, image_path, interpolation_on_activations_steps_arr,
                                                          interpolation_on_images_steps_arr, label, layers, model_name)
    heatmap = make_resize_norm(integrated_heatmaps)

    last_image = images[-1]
    t = tensor2cv(last_image)
    im, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap, use_mask=use_mask)

    return t, im, heatmap_cv, blended_img_mask, last_image, score, heatmap


def get_by_class_saliency_iig_triple(image_path,
                                     label,
                                     operations,
                                     model_name='densnet',
                                     layers=[12],
                                     interpolation_on_images_steps_arr=[0, 50],
                                     interpolation_on_activations_steps_arr=[0, 50],
                                     device='cuda',
                                     use_mask=False):
    images, integrated_heatmaps = heatmap_of_triple_iig(device, image_path,
                                                        interpolation_on_activations_steps_arr,
                                                        interpolation_on_images_steps_arr, label, layers, model_name)
    heatmap = make_resize_norm(integrated_heatmaps)

    last_image = images[-1]
    t = tensor2cv(last_image)
    im, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap, use_mask=use_mask)

    return t, im, heatmap_cv, blended_img_mask, last_image, score, heatmap


def get_by_class_saliency_iig(image_path,
                              label,
                              operations,
                              model_name='densnet',
                              layers=[12],
                              interpolation_on_images_steps_arr=[0, 50],
                              interpolation_on_activations_steps_arr=[0, 50],
                              device='cuda',
                              use_mask=False):
    images, integrated_heatmaps = heatmap_of_layer(device, image_path,
                                                   interpolation_on_activations_steps_arr,
                                                   interpolation_on_images_steps_arr, label, layers, model_name)

    heatmap = make_resize_norm(integrated_heatmaps)

    last_image = images[-1]
    t = tensor2cv(last_image)
    im, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap, use_mask=use_mask)

    return t, im, heatmap_cv, blended_img_mask, last_image, score, heatmap


def top_k_heatmap(heatmap, percent):
    top = int(IMAGE_PIXELS_COUNT * percent)
    heatmap = heatmap.reshape(IMAGE_PIXELS_COUNT)
    ind = np.argpartition(heatmap, -top)[-top:]
    all = np.arange(IMAGE_PIXELS_COUNT)
    rem = [item for item in all if item not in ind]
    heatmap[rem] = 0.
    heatmap = heatmap.reshape(224, 224)
    return heatmap


def make_resize_norm(act_grads):
    heatmap = torch.sum(act_grads.squeeze(0), dim=0)
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)

    heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().cpu().data.numpy()
    return heatmap


def heatmap_of_triple_iig(device, image_path, interpolation_on_activations_steps_arr,
                          interpolation_on_images_steps_arr,
                          label, layers, model_name):
    images, integrated_heatmaps1 = heatmap_of_triple_layers(device, image_path,
                                                            interpolation_on_activations_steps_arr,
                                                            interpolation_on_images_steps_arr, label,
                                                            layers, model_name)
    integrated_heatmaps1 = torch.tensor(make_resize_norm(integrated_heatmaps1))

    integrated_heatmaps = integrated_heatmaps1.unsqueeze(0).unsqueeze(0)

    return images, integrated_heatmaps


def heatmap_of_triple_layers(device, image_path, interpolation_on_activations_steps_arr,
                             interpolation_on_images_steps_arr,
                             label, layers, model_name):
    model = GradModel(model_name, feature_layer=layers[0])
    model.to(device)
    model.eval()
    model.zero_grad()

    images = get_images(image_path, interpolation_on_images_steps=interpolation_on_images_steps_arr[-1])

    original_image = get_images(image_path, interpolation_on_images_steps=0)

    label = torch.tensor(label, dtype=torch.long, device=device)

    original_activations = model.get_activations(original_image.to(device)).cpu()  # takes 3 sec
    activations = model.get_activations(images.to(device)).cpu()  # takes 3 sec

    activations_featmap_list = (activations.unsqueeze(1))

    if interpolation_on_activations_steps_arr[-1] > 0:

        if interpolation_on_images_steps_arr[-1] > 0:
            x, _ = torch.min(activations_featmap_list.detach(), dim=1)  # torch.Size([51, 768, 7, 7])
            basel = torch.ones_like(activations_featmap_list.detach()) * x.unsqueeze(1)
            igacts = get_interpolated_values(basel.detach(), activations_featmap_list,
                                             interpolation_on_activations_steps_arr[-1]).detach()
            # igacts = activations_featmap_list
            igacala = igacts.squeeze().detach()
            accumulated_grads = []
            for ig in igacala:
                igacts = model.get_post_activations(ig.squeeze().to(device)).detach().cpu()
                # igacts = model.get_post_activations(ig.unsqueeze(0).to(device)).cpu()
                # print(F'first is: {igacts.shape}')
                # interpolate_last_layer = True
                # if not interpolate_last_layer:
                #     model = GradModel(model_name, feature_layer=layers[0] + 1)
                #     model.to(device)
                #     model.eval()
                #     model.zero_grad()
                #     original_activations = model.get_activations(original_image.to(device)).cpu()
                #     grads = []
                #     # for act in igacts:
                #     grad = calc_grads_model_post(model, igacts, device, label,
                #                                  True).detach()
                #     # grads.append(grad)
                #     # igrads = torch.stack(grads).detach()
                #     print(grad.shape, igacts.shape)
                #
                #     integrated_heatmaps = torch.sum(
                #         (grad.squeeze()) * F.relu(original_activations),
                #         dim=[0])
                #     print(integrated_heatmaps.shape)
                #     return images, integrated_heatmaps

                x2, _ = torch.min(igacts.detach(), dim=1)  # torch.Size([51, 768, 7, 7])
                basel2 = torch.ones_like(igacts) * x2.unsqueeze(1)
                # print(F'basel2 is: {basel2.shape}')
                mega_igacts = get_interpolated_values(basel2.detach(), igacts.detach(),
                                                      interpolation_on_activations_steps_arr[-1]).detach()
                print(F'second is: {mega_igacts.shape}')

                i = 0
                model8 = GradModel(model_name, feature_layer=layers[0] + 1)
                model8.to(device)
                model8.eval()
                model8.zero_grad()
                original_activations = model8.get_activations(original_image.to(device)).detach().cpu()
                mega_grads = []
                for out_act in mega_igacts:
                    grads = []
                    for act in out_act:
                        act.requires_grad = True
                        print(F'act is: {act.shape}')
                        integrated_grads = calc_grads_model_post(model8, act.unsqueeze(0), device,
                                                                 label,
                                                                 True).detach()
                        print(F'intgrad is: {integrated_grads.shape}')
                        product_act_grads = (integrated_grads.detach().squeeze()) * F.relu(
                            original_activations.detach()).squeeze()
                        # product_act_grads = (integrated_grads.squeeze()) * F.relu(act).squeeze()
                        # print(f'product: {product_act_grads.shape}')
                        grads.append(product_act_grads.detach())
                        act = act.detach()
                        act.requires_grad = False
                    inner_grads_tensor = torch.stack(grads).detach()
                    mega_grads.append(inner_grads_tensor)
                zega_igrads = torch.stack(mega_grads).detach()
                accumulated_grads.append(zega_igrads)

            accumulation = torch.stack(accumulated_grads).detach()
            print(f'shapes: {accumulation.shape}, {zega_igrads.shape}, {inner_grads_tensor.shape}')
            zega_grads = accumulation.detach().squeeze()
            zega_acts = mega_igacts.detach().squeeze()

        # with torch.no_grad():
        #     igrads = torch.stack(grads).detach()
        #     igacts[1:] = igacts[1:] - igacts[0]
        #     # integrated_heatmaps = torch.sum(F.relu(grads.squeeze().detach()) * F.relu(igacts.squeeze()),
        #     #                                 dim=[0, 1])  # good!!!
        #     if version == 1:
        #         gradsum = torch.sum((igrads.squeeze().detach()) * (original_activations.unsqueeze(0)),
        #                             dim=[0])  # good!!!
        #     elif version == 2:
        #         gradsum = torch.sum((igrads.squeeze().detach()) * F.softplus(igacts.squeeze()),
        #                             dim=[0])  # good!!!
        #     elif version == 3:
        #         # gradsum = torch.sum(F.relu(igrads.squeeze().detach()) * F.relu(original_activations.unsqueeze(0)),
        #         #                     dim=[0])  # good!!!
        #         # gradsum = torch.sum(F.relu(igrads.squeeze().detach()) * F.relu(igacts.squeeze()),
        #         #                     dim=[0])  # good!!!
        #         gradsum = torch.sum((igrads.squeeze().detach()) * F.relu(igacts.squeeze()),
        #                             dim=[0])  # good!!!
        #
        #     integrated_heatmaps = torch.sum(gradsum, dim=[0])

        # print(f'mega igrads shape: {mega_igrads.shape}')
        # print(f'mega igacts shape: {mega_igacts.shape}')
        # integrated_heatmaps = torch.sum(
        #     (zega_igrads.squeeze()) * F.relu(original_activations.unsqueeze(0).unsqueeze(0)),
        #     dim=[0, 1, 2])
        print(zega_acts.shape, zega_grads.shape)
        integrated_heatmaps = torch.mean(
            (zega_grads.detach().squeeze()),
            dim=[0, 1, 2])
        print(integrated_heatmaps.shape)

    return images, integrated_heatmaps


def heatmap_of_layers_layer_no_interpolation(device, image_path, interpolation_on_activations_steps_arr,
                                             interpolation_on_images_steps_arr,
                                             label, layers, model_name):
    model = GradModel(model_name, feature_layer=layers[0])
    model.to(device)
    model.eval()
    model.zero_grad()

    images = get_images(image_path, interpolation_on_images_steps=0)

    original_image = get_images(image_path, interpolation_on_images_steps=0)
    label = torch.tensor(label, dtype=torch.long, device=device)
    activations = model.get_activations(images.to(device)).cpu()  # takes 3 sec
    activations_featmap_list = (activations.unsqueeze(1))
    gradients = calc_grads_model(model, activations_featmap_list, device, label).detach()
    gradients_squeeze = gradients.detach().squeeze()
    act_grads = F.relu(activations.squeeze()) * F.relu(gradients_squeeze) ** 2
    integrated_heatmaps = torch.sum(act_grads.squeeze(0), dim=0).unsqueeze(0).unsqueeze(0)
    return images, integrated_heatmaps


def heatmap_of_layer(device, image_path, interpolation_on_activations_steps_arr,
                     interpolation_on_images_steps_arr,
                     label, layers, model_name):
    images = get_images(image_path, interpolation_on_images_steps=interpolation_on_images_steps_arr[-1])
    original_image = get_images(image_path, interpolation_on_images_steps=0)

    label = torch.tensor(label, dtype=torch.long, device=device)

    original_activations = model.get_activations(original_image.to(device)).cpu()
    activations = model.get_activations(images.to(device)).cpu()

    activations_featmap_list = (activations.unsqueeze(1))

    if interpolation_on_activations_steps_arr[-1] > 0:

        if interpolation_on_images_steps_arr[-1] > 0:
            x, _ = torch.min(activations_featmap_list, dim=1)
            basel = torch.ones_like(activations_featmap_list) * x.unsqueeze(1)
            igacts = get_interpolated_values(basel.detach(), activations_featmap_list,
                                             interpolation_on_activations_steps_arr[-1]).detach()
            grads = []
            for act in igacts:
                act.requires_grad = True
                grads.append(calc_grads_model(model, (act.squeeze()), device, label).detach())
                act = act.detach()
                act.requires_grad = False

            with torch.no_grad():
                igrads = torch.stack(grads).detach()
                igacts[1:] = igacts[1:] - igacts[0]
                gradsum = torch.sum(igrads.squeeze().detach() * F.softplus(igacts.squeeze()),
                                    dim=[0])
                gradsum = torch.sum(igrads.squeeze().detach() * F.relu(igacts.squeeze()),
                                    dim=[0])
                integrated_heatmaps = torch.sum(gradsum, dim=[0])

    return images, integrated_heatmaps


def heatmap_of_layer_images(device, image_path, interpolation_on_activations_steps_arr,
                            interpolation_on_images_steps_arr,
                            label, layers, model_name):
    images = get_images(image_path, interpolation_on_images_steps=interpolation_on_images_steps_arr[-1])

    label = torch.tensor(label, dtype=torch.long, device=device)

    activations = model.get_activations(images.to(device)).cpu()

    activations_featmap_list = (activations.unsqueeze(1))

    x, _ = torch.min(activations_featmap_list, dim=1)
    grads = calc_grads(activations_featmap_list, device, label).detach()

    with torch.no_grad():
        integrated_heatmaps = torch.sum((F.relu(grads) * F.relu(activations_featmap_list)), dim=[0, 1])

    return images, integrated_heatmaps


def heatmap_of_layer_acts(device, image_path, interpolation_on_activations_steps_arr, interpolation_on_images_steps_arr,
                          label, layers, model_name):
    images = get_images(image_path, 0)

    label = torch.tensor(label, dtype=torch.long, device=device)

    activations = model.get_activations(images.to(device)).cpu()

    activations_featmap_list = (activations.unsqueeze(1))

    x, _ = torch.min(activations_featmap_list, dim=1)
    basel = torch.ones_like(activations_featmap_list) * x.unsqueeze(1)
    igacts = get_interpolated_values(basel.detach(), activations_featmap_list,
                                     interpolation_on_activations_steps_arr[-1]).detach()
    grads = calc_grads(activations_featmap_list, device, label).detach()

    with torch.no_grad():
        integrated_heatmaps = torch.sum((F.relu(grads.unsqueeze(0)) * F.relu(igacts)), dim=[0, 1])

    return images, integrated_heatmaps


def calc_grads_model_post(model, activations_featmap_list, device, label, post_feat):
    activations_gradients = backward_class_score_and_get_activation_grads(model, label, activations_featmap_list,
                                                                          only_post_features=post_feat,
                                                                          device=device)
    return activations_gradients


def calc_grads_model(model, activations_featmap_list, device, label):
    activations_gradients = backward_class_score_and_get_activation_grads(model, label, activations_featmap_list,
                                                                          only_post_features=True,
                                                                          device=device)
    return activations_gradients


def calc_grads(activations_featmap_list, device, label):
    activations_gradients = backward_class_score_and_get_activation_grads(model, label, activations_featmap_list,
                                                                          only_post_features=True,
                                                                          device=device)
    return activations_gradients


# LIFT-CAM
from captum.attr import DeepLift


class Model_Part(nn.Module):
    def __init__(self, model):
        super(Model_Part, self).__init__()
        self.model_type = None
        if model.model_str == 'convnext':
            self.avg_pool = model.avgpool
            self.classifier = model.classifier[-1]
        else:
            self.avg_pool = model.avgpool
            self.classifier = model.classifier

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def lift_cam(model, image_path, label, device, use_mask):
    images = get_images(image_path, 0)

    model.eval()
    model.zero_grad()
    output = model(images.to(device), hook=True)

    class_id = label
    if class_id is None:
        class_id = torch.argmax(output, dim=1)

    # act_map = model.get_activations(images.to(device)).detach().cpu()
    act_map = model.get_activations(images.to(device))

    model_part = Model_Part(model)
    model_part.eval()
    dl = DeepLift(model_part)
    ref_map = torch.zeros_like(act_map).to(device)
    dl_contributions = dl.attribute(act_map, ref_map, target=class_id, return_convergence_delta=False).detach()

    scores_temp = torch.sum(dl_contributions, (2, 3), keepdim=False).detach()
    scores = torch.squeeze(scores_temp, 0)
    scores = scores.cpu()

    vis_ex_map = (scores[None, :, None, None] * act_map.cpu()).sum(dim=1, keepdim=True)
    vis_ex_map = F.relu(vis_ex_map).float()

    with torch.no_grad():
        heatmap = vis_ex_map
        heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        heatmap = heatmap.squeeze().cpu().data.numpy()
        t = tensor2cv(images[-1])
        blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                           use_mask=use_mask)
    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def ig_captum(model, image_path, label, device, use_mask):
    images = get_images(image_path, 0)

    model.eval()
    model.zero_grad()
    class_id = label

    integrated_grads = captum.attr.IntegratedGradients(model)
    baseline = torch.zeros_like(images).to(device)
    attr = integrated_grads.attribute(images.to(device), baseline, class_id)

    with torch.no_grad():
        heatmap = torch.mean(attr, dim=1, keepdim=True)
        heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        heatmap = heatmap.squeeze().cpu().data.numpy()
        t = tensor2cv(images[-1])
        blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                           use_mask=use_mask)
    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def get_torchgc_model_layer(network_name, device):
    if network_name.__contains__('resnet'):
        resnet101 = torchvision.models.resnet101(pretrained=True).to(device)
        resnet101_layer = resnet101.layer4
        return resnet101, resnet101_layer
    elif network_name.__contains__('convnext'):
        convnext = torchvision.models.convnext_base(pretrained=True).to(device)
        convnext_layer = convnext.features[-1]
        return convnext, convnext_layer

    densnet201 = torchvision.models.densenet201(pretrained=True).to(device)
    densnet201_layer = densnet201.features
    return densnet201, densnet201_layer


def ablation_cam_torchcam(network_name, image_path, label, device, use_mask):
    model, layer = get_torchgc_model_layer(network_name, device)
    cam_extractor = AblationCAM(model.to(device), layer)
    images = get_images(image_path, 0)
    targets = [ClassifierOutputTarget(label)]
    hm = cam_extractor(images.to(device), targets)

    heatmap = torch.tensor(hm).unsqueeze(0)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().detach().cpu().data.numpy()

    t = tensor2cv(images[-1])
    blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                       use_mask=use_mask)

    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def fullgrad_torchcam(network_name, image_path, label, device, use_mask):
    model, layer = get_torchgc_model_layer(network_name, device)
    cam_extractor = FullGrad(model, layer)
    images = get_images(image_path, 0)
    targets = [ClassifierOutputTarget(label)]
    hm = cam_extractor(images.to(device), targets)

    heatmap = torch.tensor(hm).unsqueeze(0)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().detach().cpu().data.numpy()

    t = tensor2cv(images[-1])
    blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                       use_mask=use_mask)

    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def layercam_torchcam(network_name, image_path, label, device, use_mask):
    model, layer = get_torchgc_model_layer(network_name, device)
    cam_extractor = LayerCAM(model.to(device), layer)
    images = get_images(image_path, 0)
    targets = [ClassifierOutputTarget(label)]
    hm = cam_extractor(images.to(device), targets)

    heatmap = torch.tensor(hm).unsqueeze(0)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().detach().cpu().data.numpy()

    t = tensor2cv(images[-1])
    blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                       use_mask=use_mask)

    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def score_cam_torchcam(network_name, image_path, label, device, use_mask):
    model, layer = get_torchgc_model_layer(network_name, device)
    cam_extractor = ScoreCAM(model.to(device), layer)
    images = get_images(image_path, 0)
    targets = [ClassifierOutputTarget(label)]
    hm = cam_extractor(images.to(device), targets)
    heatmap = torch.tensor(hm).unsqueeze(0)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().detach().cpu().data.numpy()

    t = tensor2cv(images[-1])
    blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                       use_mask=use_mask)

    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def run_all_operations(model, image_path, label, model_name='densenet', device='cpu', features_layer=8,
                       operations=['iig'],
                       use_mask=False):
    results = []
    for operation in operations:
        t1, blended_img, heatmap_cv, blended_img_mask, t2, score, heatmap = run_by_class_grad(model, image_path, label,
                                                                                              model_name,
                                                                                              device,
                                                                                              features_layer,
                                                                                              operation, use_mask)
        results.append((t1, blended_img, heatmap_cv, blended_img_mask, t2, score, heatmap))
    return results


def run_by_class_grad(model, image_path, label, model_name='densenet', device='cpu', features_layer=8, operation='ours',
                      use_mask=False):
    CWD = os.getcwd()
    root = ROOT_IMAGES.format(CWD)
    print(image_path)
    img = Image.open(root + '/' + image_path).convert('RGB')
    im = preprocess(img)

    label = torch.tensor(label, dtype=torch.long, device=device)
    t1, blended_img, heatmap_cv, blended_img_mask, t2, score, heatmap = by_class_map(model, im, label,
                                                                                     operation=operation,
                                                                                     use_mask=use_mask)

    return t1, blended_img, heatmap_cv, blended_img_mask, im, score, heatmap


def calculate_max_class_of_positive_mask(image, model, percent, saliency_map):
    model.eval()
    model.zero_grad()
    heatmap = np.copy(saliency_map)
    top = int(IMAGE_PIXELS_COUNT * percent)
    heatmap = heatmap.reshape(IMAGE_PIXELS_COUNT)
    top_indexes = np.argpartition(heatmap, -top)[-top:]
    all = np.arange(IMAGE_PIXELS_COUNT)
    rem = np.delete(all, top_indexes)
    heatmap[rem] = 1.
    heatmap[top_indexes] = 0.

    if percent == 0.:
        heatmap[:] = 1.

    heatmap = heatmap.reshape(224, 224)
    img_cv = tensor2cv(image)

    heatmap_cv = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    masked_image = np.uint8((np.repeat(heatmap_cv.reshape(224, 224, 1), 3, axis=2) * img_cv))
    masked_image_tensor = preprocess(Image.fromarray(masked_image)).unsqueeze(0)
    input_predictions = model(masked_image_tensor.to(device), hook=False).detach()
    with torch.no_grad():
        probs = torch.softmax(input_predictions, dim=1)[0]
        highest_score_class = torch.max(input_predictions, 1).indices[0].item()

    return highest_score_class, probs[target_label].item(), probs[predicted_label].item()


def calculate_max_class_of_negative_mask(image, model, percent, saliency_map):
    model.eval()
    model.zero_grad()
    heatmap = np.copy(saliency_map)
    top = int(IMAGE_PIXELS_COUNT * percent)
    heatmap = heatmap.reshape(IMAGE_PIXELS_COUNT)
    negative_map = np.copy(heatmap) * -1.
    top_indexes = np.argpartition(negative_map, -top)[-top:]
    all = np.arange(IMAGE_PIXELS_COUNT)
    rem = np.delete(all, top_indexes)
    heatmap[rem] = 1.
    heatmap[top_indexes] = 0.

    if percent == 0.:
        heatmap[:] = 1.

    heatmap = heatmap.reshape(224, 224)
    img_cv = tensor2cv(image)
    heatmap_cv = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    masked_image = np.uint8((np.repeat(heatmap_cv.reshape(224, 224, 1), 3, axis=2) * img_cv))
    masked_image_tensor = preprocess(Image.fromarray(masked_image)).unsqueeze(0)
    input_predictions = model(masked_image_tensor.to(device), hook=False).detach()
    with torch.no_grad():
        probs = torch.softmax(input_predictions, dim=1)[0]
        highest_score_class = torch.max(input_predictions, 1).indices[0].item()

    return highest_score_class, probs[target_label].item(), probs[predicted_label].item()


def by_class_map(model, image, label, operation='ours', use_mask=False):
    weight_ratio = []
    model.eval()
    model.zero_grad()
    preds = model(image.unsqueeze(0).to(device), hook=True)
    _, predicted = torch.max(preds.data, 1)
    # print(f'True label {label}, predicted {predicted}')

    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    score = torch.sum(one_hot * preds)
    score.backward()
    preds.to(device)
    one_hot.to(device)
    gradients = model.get_activations_gradient()
    heatmap = grad2heatmaps(model, image.unsqueeze(0).to(device), gradients, activations=None, operation=operation,
                            score=score, do_nrm_rsz=True,
                            weight_ratio=weight_ratio)

    t = tensor2cv(image)
    blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap, use_mask=use_mask)

    return t, blended_img, heatmap_cv, blended_img_mask, t, score, heatmap


def image_show(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def calculate_perturbations_auc(model, image, saliency_map, target_label, predicted_label, positive_perturbation=True):
    # positive perturbation - from high relevance to low
    # negative perturbation - from low relevance to high
    percent_intervals = np.arange(0.0, 1.0, 0.1)
    y_target = []
    y_predicted = []
    probs_t = []
    probs_p = []
    # print(operation)
    # maxclass = int(calculate_max_class_of_positive_mask(image, model, 0.0, saliency_map))

    for percent in percent_intervals:
        if positive_perturbation:
            highest_score_class, prob_t, prob_p = calculate_max_class_of_positive_mask(image, model, percent,
                                                                                       saliency_map)
        else:
            highest_score_class, prob_t, prob_p = calculate_max_class_of_negative_mask(image, model, percent,
                                                                                       saliency_map)

        # print(f'predicted: {highest_score_class}, target was: {target_label}, max was: {maxclass}')
        probs_t.append(prob_t)
        probs_p.append(prob_p)

        if highest_score_class == target_label:
            y_target.append(1)
        else:
            y_target.append(0)

        if highest_score_class == predicted_label:
            y_predicted.append(1)
        else:
            y_predicted.append(0)
    area_under_curve_target = auc(percent_intervals, np.array(y_target))
    area_under_curve_predicted = auc(percent_intervals, np.array(y_predicted))
    area_under_curve_probs_target = auc(percent_intervals, np.array(probs_t))
    area_under_curve_probs_predicted = auc(percent_intervals, np.array(probs_p))
    # print(f'auc is: {area_under_curve_target}')
    # print(f'probs auc is: {area_under_curve_probs_target}')
    # print(f'auc is: {area_under_curve_predicted}')
    return [area_under_curve_target, area_under_curve_predicted, area_under_curve_probs_target,
            area_under_curve_probs_predicted]


def calc_score_original(input):
    global label
    preds_original_image = model(input.to(device), hook=False).detach()
    one_hot = torch.zeros(preds_original_image.shape).to(device)
    one_hot[:, label] = 1

    score_original_image = torch.sum(one_hot * preds_original_image, dim=1).detach()
    return score_original_image


def calc_score_masked(masked_image):
    global label
    preds_masked_image = model(masked_image.unsqueeze(0).to(device), hook=False).detach()
    one_hot = torch.zeros(preds_masked_image.shape).to(device)
    one_hot[:, label] = 1
    score_masked_image = torch.sum(one_hot * preds_masked_image, dim=1).detach()
    return score_masked_image


def calc_img_score(img):
    global label
    preds_masked_image = model(img.unsqueeze(0).to(device), hook=False).detach()
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
    preds_masked_image = model(img.unsqueeze(0).to(device), hook=False).detach()
    one_hot = torch.zeros(preds_masked_image.shape).to(device)
    one_hot[:, label] = 1
    score = torch.sum(one_hot * preds_masked_image, dim=1).detach()
    return score


def calc_perturbations():
    unmask = t - blended_img_mask
    sc_unmask = calc_img_score(preprocess(Image.fromarray(unmask))).item()
    current_image_results[f'PERT_{operation}'] = max(0, sc_unmask)
    return sc_unmask


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
        label = k
        label_images_paths = v
        for j, image_map in enumerate(label_images_paths):
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
        label = k
        image_paths = v
        for j, image_path in enumerate(image_paths):
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
        label = k
        label_images_paths = v
        for j, image_map in enumerate(label_images_paths):
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
        label = k
        image_paths = v
        for j, image_path in enumerate(image_paths):
            set_input.append({IMAGE_PATH: image_path, LABEL: label})
    df = pd.DataFrame(set_input)

    return df


def create_set_from_txt():
    images_by_label = {}
    with open(f'data/pics.txt') as f:
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
            set_input.append({IMAGE_PATH: image_path, LABEL: label})
    df = pd.DataFrame(set_input)

    return df


def write_results(to_write_results, input, predicted_label, target_label, save_image=False, heatmap=[], im_steps=0,
                  ac_steps=0):
    global label, score_original_image
    if to_write_results:
        im = Image.fromarray(blended_img_mask)
        masked_image = preprocess(im)
        if score_original_image == 0:
            score_original_image = calc_score_original(input)
        score_masked_image = calc_score_masked(masked_image)
        adp_pic_pert_add_percent(score_masked_image, score_original_image)

        if GRADUAL_PERTURBATION:
            perturbation_tests(heatmap, input, predicted_label, target_label)

        if USE_TOP_K:
            handle_top10_results(heatmap)

    save_mask = False
    im_to_save = blended_im
    if save_mask:
        im_to_save = blended_img_mask

    if save_image:
        title = f'method: {operation}, label: {int(label)}'
        img_dict.append({"image": im_to_save, "title": title})


# def get_blur_func():
#     klen = 11
#     ksig = 5
#     kern = insertion_deletion.gkern(klen, ksig)
#     blur = lambda x: nn.functional.conv2d(x, kern, padding=klen // 2)
#     return blur


def perturbation_tests(heatmap, input, predicted_label, target_label):
    positive_auc = calculate_perturbations_auc(model, input.squeeze(), heatmap,
                                               target_label, predicted_label,
                                               positive_perturbation=True)
    if BY_MAX_CLASS:
        current_image_results[f'PAUC_P_{operation}'] = positive_auc[1] * 100
        current_image_results[f'PAUC_PROB_P_{operation}'] = positive_auc[3] * 100
    else:
        current_image_results[f'PAUC_T_{operation}'] = positive_auc[0] * 100
        current_image_results[f'PAUC_PROB_T_{operation}'] = positive_auc[2] * 100
    negative_auc = calculate_perturbations_auc(model, input.squeeze(), heatmap,
                                               target_label, predicted_label,
                                               positive_perturbation=False)
    if BY_MAX_CLASS:
        current_image_results[f'NAUC_P_{operation}'] = negative_auc[1] * 100
        current_image_results[f'NAUC_PROB_P_{operation}'] = negative_auc[3] * 100
    else:
        current_image_results[f'NAUC_T_{operation}'] = negative_auc[0] * 100
        current_image_results[f'NAUC_PROB_T_{operation}'] = negative_auc[2] * 100


def adp_pic_pert_add_percent(score_masked_image, score_original_image):
    method_ADP = f'ADP_{operation}'
    score_original = score_original_image.item()
    score_masked = score_masked_image.item()
    adp_value = max(0, score_original - score_masked) / score_original
    current_image_results[method_ADP] = adp_value
    method_pic = f'PIC_{operation}'
    if score_original < score_masked:
        current_image_results[method_pic] = 1
    else:
        current_image_results[method_pic] = 0
    score_deleted_explanation = calc_perturbations()
    method_ADD = f'ADD_{operation}'
    add_value = (score_original - score_deleted_explanation) / score_original
    current_image_results[method_ADD] = add_value
    current_image_results[f'%_{operation}'] = (score_masked -
                                               score_original) / score_original


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


def handle_image_saving(blended_im, blended_img_mask, label, operation, save_image=False, save_mask=False):
    im_to_save = blended_im
    if save_mask:
        im_to_save = blended_img_mask

    if save_image:
        title = f'method: {operation}, label: {int(label)}'
        img_dict.append({"image": im_to_save, "title": title})


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


def write_heatmap(model_name, image_path, operation, heatmap_cv):
    CWD = os.getcwd()
    np.save("{0}/data/heatmaps/{1}_{2}_{3}".format(CWD, image_path[:-5], operation, model_name), heatmap_cv)


def write_mask(model_name, image_path, operation, masked_image):
    CWD = os.getcwd()
    np.save("{0}/data/masks/{1}_{2}_{3}".format(CWD, image_path[:-5], operation, model_name), masked_image)


ITERATION = 'iig'
models = ['densnet', 'convnext', 'resnet101']
layer_options = [12, 8]

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    results = []
    image = None

    model_name = models[2]
    FEATURE_LAYER_NUMBER = layer_options[1]

    PREV_LAYER = FEATURE_LAYER_NUMBER - 1
    interpolation_on_activations_steps_arr = [INTERPOLATION_STEPS]
    interpolation_on_images_steps_arr = [INTERPOLATION_STEPS]
    num_layers_options = [1]

    USE_MASK = True
    save_img = True
    save_heatmaps_masks = False
    to_write_results = False
    operations = ['iig', 'fullgrad', 'ablation-cam', 'lift-cam', 'layercam', 'ig',
                  'gradcam', 'gradcampp', 'x-gradcam']
    operations = ['iig', 'iig-ablation-im', 'iig-ablation-ac', 'gradcam']
    operations = ['iig', 'lift-cam', 'ablation-cam', 'gradcam', 'gradcampp']
    operations = ['iig']

    torch.nn.modules.activation.ReLU.forward = ReLU.forward
    if model_name.__contains__('vgg'):
        torch.nn.modules.activation.ReLU.forward = ReLU.forward

    model = GradModel(model_name, feature_layer=FEATURE_LAYER_NUMBER)
    model.to(device)
    model.eval()
    model.zero_grad()

    # df = coco_calculate_localization()
    # IS_COCO_BBOX = True
    # df = voc_calculate_localization()
    # IS_VOC_BBOX = True
    # df = create_set_from_voc()
    # IS_VOC = True
    # df = create_set_from_coco()
    # IS_COCO = True
    df = create_set_from_txt()
    print(len(df))
    df_len = len(df)

    for index, row in tqdm(df.iterrows()):

        current_image_results = {}
        image_path = row[IMAGE_PATH]
        label = row[LABEL]
        target_label = label
        input = get_images(image_path, 0)
        input_predictions = model(input.to(device), hook=False).detach()
        predicted_label = torch.max(input_predictions, 1).indices[0].item()

        if BY_MAX_CLASS:
            label = predicted_label

        res_class_saliency = run_all_operations(model, image_path=image_path,
                                                label=label, model_name=model_name, device=device,
                                                features_layer=FEATURE_LAYER_NUMBER,
                                                operations=operations[1:], use_mask=USE_MASK)

        operation_index = 0
        score_original_image = 0
        img_dict = []
        for operation in operations:
            if operation == 'iig':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = \
                    get_by_class_saliency_iig(image_path=image_path,
                                              label=label,
                                              operations=['iig'],
                                              model_name=model_name,
                                              layers=[FEATURE_LAYER_NUMBER],
                                              interpolation_on_images_steps_arr=interpolation_on_images_steps_arr,
                                              interpolation_on_activations_steps_arr=interpolation_on_activations_steps_arr,
                                              device=device,
                                              use_mask=USE_MASK)
            elif operation == 'iig-ablation-ac':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = \
                    get_by_class_saliency_iig_ablation_study_ac(
                        image_path=image_path,
                        label=label,
                        operations=['iig'],
                        model_name=model_name,
                        layers=[FEATURE_LAYER_NUMBER],
                        interpolation_on_images_steps_arr=interpolation_on_images_steps_arr,
                        interpolation_on_activations_steps_arr=interpolation_on_activations_steps_arr,
                        device=device,
                        use_mask=USE_MASK)
            elif operation == 'iig-ablation-im':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = \
                    get_by_class_saliency_iig_ablation_study_im(
                        image_path=image_path,
                        label=label,
                        operations=['iig'],
                        model_name=model_name,
                        layers=[FEATURE_LAYER_NUMBER],
                        interpolation_on_images_steps_arr=interpolation_on_images_steps_arr,
                        interpolation_on_activations_steps_arr=interpolation_on_activations_steps_arr,
                        device=device,
                        use_mask=USE_MASK)
            elif operation == 'iig-triple':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = \
                    get_by_class_saliency_iig_triple(image_path=image_path,
                                                     label=label,
                                                     operations=['iig'],
                                                     model_name=model_name,
                                                     layers=[PREV_LAYER],
                                                     interpolation_on_images_steps_arr=interpolation_on_images_steps_arr,
                                                     interpolation_on_activations_steps_arr=interpolation_on_activations_steps_arr,
                                                     device=device,
                                                     use_mask=USE_MASK)
            elif operation == 'lift-cam':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = lift_cam(
                    model,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
            elif operation == 'score-cam':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = score_cam_torchcam(
                    model_name,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
            elif operation == 'ablation-cam':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = ablation_cam_torchcam(
                    model_name,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
            elif operation == 'ig':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = ig_captum(
                    model,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
            elif operation == 'blurig':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = blurig(
                    model,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
            elif operation == 'guidedig':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = guidedig(
                    model,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
            elif operation == 'layercam':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = layercam_torchcam(
                    model_name,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
            elif operation == 'fullgrad':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = fullgrad_torchcam(
                    model_name,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
            else:
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = res_class_saliency[
                    operation_index]
                operation_index = operation_index + 1
                # write_results(input, predicted_label, target_label, save_image=save_img, heatmap=heatmap)
            # CWD = os.getcwd()
            # heatmap = np.load("{0}/data/heatmaps/{1}_{2}.npy".format(CWD, image_path[:-5], operation))
            # image_show(heatmap, 'show')

            evaluations.run_all_evaluations(input, operation, predicted_label, target_label,
                                            save_image=save_img,
                                            heatmap=heatmap,
                                            blended_img_mask=blended_img_mask, blended_im=None, t=t,
                                            model=model,
                                            result_dict=current_image_results)
            handle_image_saving(blended_im, blended_img_mask, label, operation, save_image=True, save_mask=False)
            # write_results(to_write_results, input, predicted_label, target_label, save_image=save_img, heatmap=heatmap)
            if save_heatmaps_masks:
                write_heatmap(model_name, image_path, operation, heatmap_cv)
                write_mask(model_name, image_path, operation, blended_img_mask)

        if save_img:
            string_lbl = label_map.get(int(label))
            if IS_VOC:
                string_lbl = voc_classes[int(label)]
            if IS_COCO:
                string_lbl = coco_label_list[int(label)]
            write_imgs_iterate(f'qualitive_results/{model_name}_{string_lbl}_{image_path}')
        score_original_image = calc_score_original(input)

        current_image_results[IMAGE_PATH] = image_path
        current_image_results[LABEL] = label
        current_image_results[INPUT_SCORE] = score_original_image.item()

        results.append(current_image_results)
        torch.cuda.empty_cache()
        if index % 100 == 0:
            if to_write_results:
                save_results_to_csv_step(index)
            print(index)
    if to_write_results:
        save_results_to_csv()
