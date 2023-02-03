from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Union, Dict, List, Tuple
from torchvision.transforms import transforms
from tqdm import tqdm
from transformers import ViTForImageClassification
# from config import config
# from pytorch_lightning import seed_everything
from torch.nn import functional as F
import numpy as np
import torch
from enum import Enum
from iig.vit_saliency_map_generator import BY_MAX_CLASS
from iig.evaluation_metrics.seg_cls_perturbation_tests_new import eval_perturbation_test

IS_CONVNET = False


def run_all_evaluations(input, operation, predicted_label, target_label, save_image=False, heatmap=[],
                        blended_img_mask=None,
                        blended_im=None, t=None, model=None, result_dict=None):
    model.zero_grad()
    images_and_masks = [
        {'image_resized': input.cuda(), 'image_mask': torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).cuda()}]
    gt_classes_list = []
    if BY_MAX_CLASS:
        gt_classes_list.append(predicted_label)
    else:
        gt_classes_list.append(target_label)
    pic, adp, add = infer_adp_pic_add(model_for_image_classification=model,
                                      images_and_masks=images_and_masks,
                                      gt_classes_list=gt_classes_list,
                                      is_convnet=IS_CONVNET,
                                      )
    method_pic = f'PIC_{operation}'
    result_dict[method_pic] = pic
    method_adp = f'ADP_{operation}'
    result_dict[method_adp] = adp
    method_add = f'ADD_{operation}'
    result_dict[method_add] = add

    auc_perturbation_list1, auc_deletion_insertion_list1 = infer_perturbation_tests(
        images_and_masks=images_and_masks,
        model_for_image_classification=model,
        perturbation_type='POS',
        is_calculate_deletion_insertion=True,
        gt_classes_list=gt_classes_list,
        is_explainee_convnet=IS_CONVNET,
    )
    auc_perturbation1, auc_deletion_insertion1 = np.mean(auc_perturbation_list1), np.mean(auc_deletion_insertion_list1)
    auc_perturbation_list2, auc_deletion_insertion_list2 = infer_perturbation_tests(
        images_and_masks=images_and_masks,
        model_for_image_classification=model,
        perturbation_type='NEG',
        is_calculate_deletion_insertion=True,
        gt_classes_list=gt_classes_list,
        is_explainee_convnet=IS_CONVNET,
    )
    auc_perturbation2, auc_deletion_insertion2 = np.mean(auc_perturbation_list2), np.mean(auc_deletion_insertion_list2)
    method_neg = f'NEG_{operation}'
    result_dict[method_neg] = auc_perturbation2
    method_pos = f'POS_{operation}'
    result_dict[method_pos] = auc_perturbation1
    method_insert = f'INS_{operation}'
    result_dict[method_insert] = auc_deletion_insertion2
    method_delete = f'DEL_{operation}'
    result_dict[method_delete] = auc_deletion_insertion1


def infer_perturbation_tests(images_and_masks,
                             model_for_image_classification,
                             perturbation_type, is_calculate_deletion_insertion,
                             gt_classes_list: List[int],
                             is_explainee_convnet: bool,
                             ) -> Tuple[List[float], List[float]]:
    """
    :param config: contains the configuration of the perturbation test:
        * neg: True / False
    """
    aucs_perturbation = []
    aucs_auc_deletion_insertion = []
    # for image_idx, image_and_mask in tqdm(enumerate(images_and_masks)):
    for image_idx, image_and_mask in enumerate(images_and_masks):
        image, mask = image_and_mask["image_resized"], image_and_mask["image_mask"]  # [1,3,224,224], [1,1,224,224]
        outputs = [
            {'image_resized': image,
             'image_mask': mask,
             'target_class': torch.tensor([gt_classes_list[image_idx]]),
             }
        ]
        auc_perturbation, auc_deletion_insertion = eval_perturbation_test(experiment_dir=Path(""),
                                                                          model=model_for_image_classification,
                                                                          outputs=outputs,
                                                                          perturbation_type=perturbation_type,
                                                                          is_calculate_deletion_insertion=is_calculate_deletion_insertion,
                                                                          is_convenet=is_explainee_convnet,
                                                                          )
        aucs_perturbation.append(auc_perturbation)
        aucs_auc_deletion_insertion.append(auc_deletion_insertion)
    return aucs_perturbation, aucs_auc_deletion_insertion


def infer_adp_pic_add(model_for_image_classification: ViTForImageClassification,
                      images_and_masks,
                      gt_classes_list: List[int],
                      is_convnet: bool,
                      ):
    adp_values, pic_values, add_values = [], [], []

    # for image_idx, image_and_mask in tqdm(enumerate(images_and_masks), total=len(gt_classes_list)):
    image_idx = 0
    for image_and_mask in images_and_masks:
        image, mask = image_and_mask["image_resized"], image_and_mask["image_mask"]  # [1,3,224,224], [1,1,224,224]
        normalize_mean, normalize_std = get_normalization_mean_std(is_convnet=is_convnet)

        norm_original_image = normalize(image.clone(), mean=normalize_mean, std=normalize_std)
        scattered_image = scatter_image_by_mask(image=image, mask=mask)
        norm_scattered_image = normalize(scattered_image.clone(), mean=normalize_mean, std=normalize_std)
        black_scattered_image = image * (1 - mask)
        norm_black_scattered_image = normalize(black_scattered_image.clone(), mean=normalize_mean, std=normalize_std)
        metrics = run_evaluation_metrics(model_for_image_classification=model_for_image_classification,
                                         inputs=norm_original_image,
                                         inputs_scatter=norm_scattered_image,
                                         inputs_black_scatter=norm_black_scattered_image,
                                         gt_class=gt_classes_list[image_idx],
                                         is_explainee_convnet=is_convnet,
                                         )
        adp_values.append(metrics["avg_drop_percentage"])
        pic_values.append(metrics["percentage_increase_in_confidence_indicators"])
        add_values.append(metrics["avg_drop_in_deletion_percentage"])
        image_idx = image_idx + 1

    averaged_drop_percentage = 100 * np.mean(adp_values)
    percentage_increase_in_confidence = 100 * np.mean(pic_values)
    avg_drop_in_deletion_percentage = 100 * np.mean(add_values)

    return percentage_increase_in_confidence, averaged_drop_percentage, avg_drop_in_deletion_percentage


def get_normalization_mean_std(is_convnet: bool) -> Tuple[List[float], List[float]]:
    mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if is_convnet else (
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5])
    return mean, std


def normalize2(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    return tensor


def scatter_image_by_mask(image, mask):
    return image * mask


def calculate_average_change_percentage(full_image_confidence: float,
                                        saliency_map_confidence: float,
                                        ) -> float:
    """
    Higher is better
    """
    return (saliency_map_confidence - full_image_confidence) / full_image_confidence


def calculate_avg_drop_percentage(full_image_confidence: float,
                                  saliency_map_confidence: float,
                                  ) -> float:
    """
    Lower is better
    """
    return max(0, full_image_confidence - saliency_map_confidence) / full_image_confidence


def calculate_percentage_increase_in_confidence(full_image_confidence: float,
                                                saliency_map_confidence: float,
                                                ) -> float:
    """
    Higher is better
    """
    return 1 if full_image_confidence < saliency_map_confidence else 0


def calculate_avg_drop_in_deletion_percentage(full_image_confidence: float,
                                              black_saliency_map_confidence: float,
                                              ) -> float:
    """
    Higher is better
    """
    return max(0, full_image_confidence - black_saliency_map_confidence) / full_image_confidence


def run_evaluation_metrics(model_for_image_classification,
                           inputs,
                           inputs_scatter,
                           inputs_black_scatter,
                           gt_class: int,
                           is_explainee_convnet: bool
                           ):
    full_image_probability_by_index = get_probability_and_class_idx_by_index(
        logits=model_for_image_classification(inputs) if is_explainee_convnet else model_for_image_classification(
            inputs),
        index=gt_class)
    saliency_map_probability_by_index = get_probability_and_class_idx_by_index(
        logits=model_for_image_classification(
            inputs_scatter) if is_explainee_convnet else model_for_image_classification(
            inputs_scatter),
        index=gt_class)
    black_saliency_map_probability_by_index = get_probability_and_class_idx_by_index(
        logits=model_for_image_classification(
            inputs_black_scatter) if is_explainee_convnet else model_for_image_classification(
            inputs_black_scatter),
        index=gt_class)
    avg_drop_percentage = calculate_avg_drop_percentage(
        full_image_confidence=full_image_probability_by_index,
        saliency_map_confidence=saliency_map_probability_by_index)

    percentage_increase_in_confidence_indicators = calculate_percentage_increase_in_confidence(
        full_image_confidence=full_image_probability_by_index,
        saliency_map_confidence=saliency_map_probability_by_index
    )

    avg_drop_in_deletion_percentage = calculate_avg_drop_in_deletion_percentage(
        full_image_confidence=full_image_probability_by_index,
        black_saliency_map_confidence=black_saliency_map_probability_by_index
    )

    return dict(avg_drop_percentage=avg_drop_percentage,
                percentage_increase_in_confidence_indicators=percentage_increase_in_confidence_indicators,
                avg_drop_in_deletion_percentage=avg_drop_in_deletion_percentage
                )


def get_probability_and_class_idx_by_index(logits, index: int) -> float:
    probability_distribution = F.softmax(logits[0], dim=-1)
    predicted_probability_by_idx = probability_distribution[index].item()
    return predicted_probability_by_idx


def plot_image(image, title=None) -> None:  # [1,3,224,224] or [3,224,224]
    image = image if len(image.shape) == 3 else image.squeeze(0)
    plt.imshow(image.cpu().detach().permute(1, 2, 0))
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()
