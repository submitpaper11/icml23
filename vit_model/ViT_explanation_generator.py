import argparse
import torch
import numpy as np
from numpy import *

ITERATION_STEPS = 5


# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                    for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer + 1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention


class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
        output = self.model(input)
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)
        # output = output.logits  # todo: My addition
        # one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32).to("cuda")
        one_hot = torch.zeros(1, output.size()[-1]).to('cuda')
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        # one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        return self.model.relprop(one_hot_vector.to(input.device), method=method, is_ablation=is_ablation,
                                  start_layer=start_layer, **kwargs)
        # return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation,
        #                           start_layer=start_layer, **kwargs)

    def generate_IIG(self, images, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
        # print(self.model)
        # print(list(self.model.children())[1:])
        # print(list(list(self.model.children())[1:][-4:]))

        total = []
        only_integrated_attention = []
        for input in images:
            output = self.model(input)
            kwargs = {"alpha": 1}
            if index == None:
                index = np.argmax(output.cpu().data.numpy(), axis=-1)
            # output = output.logits  # todo: My addition
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0, index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            rollout_attention = self.model.relprop(torch.tensor(one_hot_vector).to(input.device),
                                                   method="iig_rollout", is_ablation=is_ablation,
                                                   start_layer=start_layer, **kwargs).detach()
            only_integrated_attention.append(rollout_attention[:, 0, 1:].reshape(1, 1, 14, 14))
            interpolated_attention = get_interpolated_values(torch.zeros_like(rollout_attention.cpu()),
                                                             rollout_attention.cpu(),
                                                             num_steps=ITERATION_STEPS).squeeze(1)

            interpolated_attention = torch.functional.F.interpolate(interpolated_attention, size=(768),
                                                                    mode='linear',
                                                                    align_corners=False)
            print(f'iga : {interpolated_attention.shape}')
            all = []
            for attn in interpolated_attention:
                attn = attn.cuda()
                # classifier = torch.nn.Sequential(*list(self.model.children())[-4:])
                encoder_last = list(list(self.model.children())[1:][0])[0]
                # encoder_last = torch.nn.Sequential(*list(list(self.model.children())[1:][0])[0:10])
                output = encoder_last(attn.unsqueeze(0))
                output = self.model.norm(output)
                output = self.model.pool(output, dim=1, indices=torch.tensor(0, device=output.device))
                output = output.squeeze(1)
                output = self.model.head(output)

                kwargs = {"alpha": 1}
                if index == None:
                    index = np.argmax(output.cpu().data.numpy(), axis=-1)
                # output = output.logits  # todo: My addition
                one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                one_hot[0, index] = 1
                one_hot_vector = one_hot
                one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                one_hot = torch.sum(one_hot.cuda() * output)
                self.model.zero_grad()
                one_hot.backward(retain_graph=True)
                rollout_attention = self.model.relprop(torch.tensor(one_hot_vector).to(attn.device),
                                                       method="gradients", is_ablation=is_ablation,
                                                       start_layer=start_layer, **kwargs).reshape(1, 1, 14, 14).detach()
                all.append(rollout_attention)
            total.append(torch.stack(all))
        with torch.no_grad():
            integrated_attn = torch.stack(only_integrated_attention)
            total = torch.stack(total)
            print(total.shape, integrated_attn.shape)
            x = torch.sum(total * integrated_attn.unsqueeze(1), dim=[0, 1])
        return x


def get_interpolated_values(baseline, target, num_steps):
    """this function returns a list of all the images interpolation steps."""
    if num_steps <= 0: return np.array([])
    if num_steps == 1: return np.array([baseline, target])

    delta = target - baseline

    if baseline.ndim == 3:
        scales = np.linspace(0, 1, num_steps + 1, dtype=np.float32)[:, np.newaxis, np.newaxis,
                 np.newaxis]  # newaxis = unsqueeze
    elif baseline.ndim == 4:
        scales = np.linspace(0, 1, num_steps + 1, dtype=np.float32)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    elif baseline.ndim == 5:
        scales = np.linspace(0, 1, num_steps + 1, dtype=np.float32)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis,
                 np.newaxis]
    elif baseline.ndim == 6:
        scales = np.linspace(0, 1, num_steps + 1, dtype=np.float32)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis,
                 np.newaxis, np.newaxis]

    shape = (num_steps + 1,) + delta.shape
    deltas = scales * np.broadcast_to(delta.detach().numpy(), shape)
    interpolated_activations = baseline + deltas

    return interpolated_activations


class Baselines:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_cam_attn(self, input, index=None):
        output = self.model(input.cuda(), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        #################### attn
        grad = self.model.blocks[-1].attn.get_attn_gradients()
        cam = self.model.blocks[-1].attn.get_attention_map()
        cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam
        #################### attn

    def generate_rollout(self, input, start_layer=0):
        self.model(input)
        blocks = self.model.blocks
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attn.get_attention_map()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        return rollout[:, 0, 1:]

    def generate_rollout_grads(self, input, start_layer=0):
        self.model(input)
        blocks = self.model.blocks
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attn.get_attention_map()
            attn_grads = blk.attn.get_attn_gradients()
            print(attn_heads.shape)
            attn_heads = attn_heads * attn_grads
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        return rollout[:, 0, 1:]

    def generate_iig(self, input, index=None):
        self.do_backward(index, input)
        #################### attn
        # grad = self.model.blocks[-1].attn.get_attn_gradients()
        cam = self.model.blocks[-1].attn.get_attention_map()
        print(f'shape is : {cam.shape}')
        cam = torch.mean(cam, dim=1).unsqueeze(1).unsqueeze(1)
        resized_cam = torch.nn.functional.interpolate(cam.cuda(), size=(3, 224, 224), mode='trilinear',
                                                      align_corners=False).squeeze(1)
        self.do_backward(index, resized_cam)
        grad = self.model.blocks[-1].attn.get_attn_gradients()
        cam = self.model.blocks[-1].attn.get_attention_map()
        cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
        print(f'shape is : {grad.shape}')
        grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
        # grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (grad).mean(0)

        # cam = cam.mean(0).clamp(min=0)  # I added
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def do_backward(self, index, input):
        output = self.model(input.cuda(), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
