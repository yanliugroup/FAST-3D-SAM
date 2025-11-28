from dataset.datasets import load_data_volume_dataset
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
import numpy as np
import logging
import sys
from monai.losses import DiceCELoss, DiceLoss
import ast
import torch.nn.functional as F

import torch

from utils.click_encoding import DistMaps
import torch.nn as nn
from functools import partial
import os
from utils.util import setup_logger, count_parameters
import time
from utils.click_encoding import DistMaps
from monai.losses import DiceCELoss, DiceLoss
import pygame
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from utils.ui_utils import preprocess_ct_image, Button
pygame.init()

def build_network_and_data():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", default=None, type=str, choices=["cuhk", "baidu_new", "baidu_XR", "lora_incep", "tri_attn", "tri_attn_loraAdapter_pEncodeS_miniDe"]
    )
    parser.add_argument(
        "--pretrained", action="store_true"
    )
    parser.add_argument(
        "--data", default=None, type=str
    )
    parser.add_argument(
        "--snapshot_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--load_weight",
        default="original",
        type=str,
    )
    parser.add_argument(
        "--user_name",
        default="Default",
        type=str,
    )
    parser.add_argument(
        "--data_prefix",
        default="",
        type=str,
    )
    parser.add_argument(
        "--rand_crop_size",
        default=0,
        nargs='+', type=int,
    )
    parser.add_argument(
        "--screen_size",
        default=1,
        type=float,
    )
    parser.add_argument(
        "--input_image_size",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
    )
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--max_epoch", default=100, type=int)
    parser.add_argument("--eval_interval", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_worker", default=12, type=int)
    parser.add_argument("-tolerance", default=5, type=int)

    args = parser.parse_args()
    
    args.method = "tri_attn_loraAdapter_pEncodeS_miniDe"
    args.snapshot_path = "exps/tri_attn_loraAdapter_pEncodeS_miniDe_128_bs2_10Click_simpleClick_original-weight_norm/"

    if args.method == "cuhk":
        from modeling.Med_SAM.image_encoder_sam import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
    elif args.method == "baidu_new":
        from modeling.Med_SAM.image_encoder_baidu_new import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
        from modeling.Med_SAM.mask_decoder_sam import MaskDecoder
        from modeling.Med_SAM.prompt_encoder_sam import PromptEncoder, TwoWayTransformer
    elif args.method == "baidu_XR":
        from modeling.Med_SAM.image_encoder_baidu_simple import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
        from modeling.Med_SAM.mask_decoder_mamba import MaskDecoder
        from modeling.Med_SAM.prompt_encoder_simple import PromptEncoderS
    elif args.method == "lora_incep":
        from modeling.Med_SAM.image_encoder_lora_incep_sam import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
        import loralib as lora
    elif args.method == "tri_attn":
        from modeling.Med_SAM.image_encoder_tri_attn import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
        import loralib as lora
    elif args.method == "tri_attn_loraAdapter_pEncodeS_miniDe":
        from modeling.Med_SAM.image_encoder_tri_attn_Lora_Adapter import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
        import loralib as lora
        from modeling.Med_SAM.mask_decoder_mamba import MaskDecoder
        from modeling.Med_SAM.prompt_encoder_simple import PromptEncoderS
    else:
        raise "unknown method"
    
    assert args.user_name != "Default"
    file_path = "utils/all_users.txt"
    previous_user_names = []

    ids_to_names = ["Default" for i in range(100)]
    names_to_id = {}
    names_to_target = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            info = line.rstrip('\n')
            if "ID" not in info:
                continue
            name = info.split(' - ID:')[0]
            id = info.split(' - ID:')[1].split(' - CurrentTargetIdx:')[0]
            current_target_idx = info.split(' - CurrentTargetIdx:')[1]
            id = int(id)
            previous_user_names.append(name)
            ids_to_names[id] = name
            names_to_id[name] = id
            names_to_target[name] = current_target_idx
    
    ids_to_names = [p for p in ids_to_names if p != "Default"]

    if args.user_name in names_to_id:
        args.user_id = names_to_id[args.user_name]
    else:
        args.user_id = len(ids_to_names) + 1
        current_target_idx = 0
        names_to_target[args.user_name] = 0

    if args.user_name not in previous_user_names:
        print("Add new user:", args.user_name)
        previous_user_names.append(args.user_name)
        names_to_id[args.user_name] = args.user_id
    with open(file_path, 'w') as file:
        for name in previous_user_names:
            file.writelines(name+" - ID:"+str(names_to_id[name]) + " - CurrentTargetIdx:" + str(names_to_target[name]) + "\n")

    device = f"cuda:{args.user_id % 4}"
    args.device = device
    
    args.current_target_idx = names_to_target[args.user_name]

    if args.rand_crop_size == 0:
        if args.data in ["kits", "colon"]:
            args.rand_crop_size = (256, 256, 256)
        if args.data in ["pancreas", "lits", "brain", "hepatic", "kits23"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    args.snapshot_path = os.path.join(args.snapshot_path, "['lung','lung2','Lung421','pancreas','kits23','hepatic']")
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    folder_path = os.path.join("exp_iseg_output", args.user_name)

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        print("文件夹已创建")
    else:
        print("路径已存在，并且是一个文件夹")
    
    setup_logger(logger_name="test", root=folder_path, screen=True, tofile=True)
    logger = logging.getLogger(f"test")
    logger.info(str(args))

    args.data = ['lung', 'lung2', 'Lung421']
    args.data_prefix = []
    for dataset_name in args.data:
        args.data_prefix.append(f"../datafile/{dataset_name}_Test0327")
    print(args.data_prefix)

    val_data = load_data_volume_dataset(
        datas=args.data,
        path_prefixes=args.data_prefix,
        batch_size=2,
        augmentation=False,
        split="test",
        deterministic=True,
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker = args.num_worker
    )
    if args.load_weight=="original":
        sam = sam_model_registry["vit_b"](checkpoint="ckpt/sam_vit_b_01ec64.pth")
    elif args.load_weight=="medsam":
        sam = sam_model_registry["vit_b"](checkpoint="ckpt/medsam_vit_b.pth")
    else:
        raise "Unknown pretrain weight."
    logger.info(f'Using pretrained weight: {args.load_weight}')

    mask_generator = SamAutomaticMaskGenerator(sam)
    img_encoder = ImageEncoderViT_3d(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        cubic_window_size=8,
        out_chans=256,
        num_slice = 16)

    load_pretrained = True
    if args.method == "tri_attn_loraAdapter_pEncodeS_miniDe":
        file = "best_debug.pth.tar"
    elif args.method == "baidu_new":
        file = "best_debug.pth.tar"
    elif args.method == "baidu_XR":
        file = "best_debug.pth.tar"
    
    for i in img_encoder.blocks:
        i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)
    
    print("load_pretrained", load_pretrained)
    if not load_pretrained:
        img_encoder.load_state_dict(mask_generator.predictor.model.image_encoder.state_dict(), strict=True)
    else:
        img_encoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["encoder_dict"], strict=True)
    del sam
    img_encoder.to(device)

    if args.method == "tri_attn_loraAdapter_pEncodeS_miniDe":
        prompt_encoder = PromptEncoderS(32)
        mask_decoder = MaskDecoder()
    elif args.method == "baidu_new":
        prompt_encoder = PromptEncoder(256, [16, 16, 16], [128, 128, 128], mask_in_chans=16)
        two_way = TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048)
        two_way.load_state_dict(mask_generator.predictor.model.mask_decoder.transformer.state_dict(), strict=True)
        mask_decoder = MaskDecoder(transformer_dim=256, transformer=two_way, num_multimask_outputs=1)
    elif args.method == "baidu_XR":
        prompt_encoder = PromptEncoderS(32)
        mask_decoder = MaskDecoder()
    if load_pretrained:
        prompt_encoder.load_state_dict(
            torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["feature_dict"], strict=True)
        mask_decoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["decoder_dict"],
                            strict=True)
    
    prompt_encoder.to(device)
    mask_decoder.to(device)

    logger.info(f'The img_encoder model has {count_parameters(img_encoder):,}M trainable parameters.')
    logger.info(f'The mask_decoder model has {count_parameters(mask_decoder):,}M trainable parameters.')

    img_encoder.eval()
    prompt_encoder.eval()
    mask_decoder.eval()
    
    return val_data, [img_encoder, prompt_encoder, mask_decoder], args, logger


def ord_screen_to_model(point_screen, image_size):
    screen_x, screen_y = point_screen
    return image_size - screen_y - 1, screen_x

def get_clip_info(ct_image_shape, x, y, current_image_index, patch_size):
    z = current_image_index
    
    d_min = z - patch_size//2
    d_max = z + patch_size//2
    h_min = x - patch_size//2
    h_max = x + patch_size//2
    w_min = y - patch_size//2
    w_max = y + patch_size//2
    d_l = max(0, -d_min)
    d_r = max(0, d_max - ct_image_shape[0])
    h_l = max(0, -h_min)
    h_r = max(0, h_max - ct_image_shape[1])
    w_l = max(0, -w_min)
    w_r = max(0, w_max - ct_image_shape[2])

    d_min = max(0, d_min)
    h_min = max(0, h_min)
    w_min = max(0, w_min)
    
    clip_position = {
                    "min_max": [d_min, d_max, h_min, h_max, w_min, w_max],
                    "padding": (w_l, w_r, h_l, h_r, d_l, d_r)
                }
    return clip_position



def generate_mask_by_click(points_pos, points_neg, img, seg, name, networks, args, logger):
    img_encoder, prompt_encoder, mask_decoder = networks
    
    input_image_size = args.input_image_size
    device = args.device
    
    patch_size = args.rand_crop_size[0]
    
    loss_overview = {
        "kitney": [],
        "lung": [],
        "pancreas": [],
        "hepaticvessel": []
    }
    mean_overview = {
        "kitney": [],
        "lung": [],
        "pancreas": [],
        "hepaticvessel": []
    }
    
    dis_map = DistMaps(2, use_disks=True)
    
    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    
    with torch.no_grad():
        loss_summary = []

        path = name
        
        idx = test_idx
        
        # print('seg: ', seg.sum())
        out = F.interpolate(img.float(), scale_factor=input_image_size / patch_size, mode='trilinear')  # TODO: for 256
        input_batch = out.to(device)
        batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
        input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)
        
        seg = seg.to(device)
        
        points_pos = torch.tensor([
            [54, 71, 70],
            [57, 68, 65],
            [63, 74, 55],
            [52, 72, 74],
            [59, 82, 71]
        ])
        
        points_neg = torch.tensor([
            [7, 112, 5]
        ])
        
        points_pos, points_neg = points_pos.to(device), points_neg.to(device)
        
        # randomly sample click points
        this_batch_points_feature = []
        for _seg in seg:
            _seg = _seg.unsqueeze(0)
            assert _seg.shape[-1] == 128
            
            l = len(torch.where(_seg == 1)[0])
            if l > 0:
                positive_feat = dis_map.get_coord_features(points_pos, 1, 128, 128, 128)
            else:
                print("no target")
                positive_feat = torch.zeros(1, 1, 128, 128, 128).to(device)

            l = len(torch.where(_seg == 0)[0])
            negative_feat = dis_map.get_coord_features(points_neg, 1, 128, 128, 128)
            
            print(points_pos, points_neg)
            
            this_batch_points_feature.append(
                torch.cat([positive_feat, negative_feat], dim=1)
            )

        prompt_input = torch.cat(this_batch_points_feature, 0).float()
        
        
        point_feature = prompt_encoder(prompt_input)
        batch_features = img_encoder(input_batch, batchsize, point_feature)  # bs, C, H, W, D
        
        masks = mask_decoder(batch_features)
        masks = masks.permute(0, 1, 4, 2, 3)  # bs, C, D, H, W
    
        loss = dice_loss(masks, seg)
        loss_summary.append(loss.detach().cpu().numpy())
        #print(path)
        
        if "kits" in path[0]:
            loss_overview["kitney"].append(loss.mean().detach().cpu().numpy())
            mean_overview["kitney"].append(img.mean().detach().cpu().numpy().squeeze())
        elif "lung" in path[0].lower():
            loss_overview["lung"].append(loss.mean().detach().cpu().numpy())
            mean_overview["lung"].append(img.mean().detach().cpu().numpy().squeeze())
        elif "pancreas" in path[0]:
            loss_overview["pancreas"].append(loss.mean().detach().cpu().numpy())
            mean_overview["pancreas"].append(img.mean().detach().cpu().numpy().squeeze())
        elif "hepaticvessel" in path[0]:
            loss_overview["hepaticvessel"].append(loss.mean().detach().cpu().numpy())
            mean_overview["hepaticvessel"].append(img.mean().detach().cpu().numpy().squeeze())
        
    logger.info("- Val metrics: " + str(np.mean(loss_summary)))

    return_mask = torch.nn.functional.interpolate(masks.float(), scale_factor=(1, 2, 2), mode='trilinear')
    return_mask = F.softmax(return_mask, dim=1)[:,1]
    print("预测最大概率", return_mask.max())
    return_mask = return_mask > 0.5
    return return_mask.squeeze().bool()


def crop_by_click_position(img, seg, center_pos, patch_size=128, crop_depth=True):
    
    if len(img.shape) == 5:
        img = img.squeeze()
    if img.shape[0] == 3:
        img = img[0]
    if len(seg.shape) > 3:
        seg = seg.squeeze()
    assert len(img.shape) == 3
    assert len(seg.shape) == 3
    
    x, y, z = center_pos
    
    d_min = z - patch_size//2
    d_max = z + patch_size//2
    h_min = x - patch_size//2
    h_max = x + patch_size//2
    w_min = y - patch_size//2
    w_max = y + patch_size//2
    d_l = max(0, -d_min)
    d_r = max(0, d_max - img.shape[0])
    h_l = max(0, -h_min)
    h_r = max(0, h_max - img.shape[1])
    w_l = max(0, -w_min)
    w_r = max(0, w_max - img.shape[2])

    d_min = max(0, d_min)
    h_min = max(0, h_min)
    w_min = max(0, w_min)
    if crop_depth:
        clip_image = img[None, None, d_min:d_max, h_min:h_max, w_min:w_max].clone().float()
        clip_image = F.pad(clip_image, (w_l, w_r, h_l, h_r, d_l, d_r)) # NOTE: check why?
    else:
        clip_image = img[None, None, :, h_min:h_max, w_min:w_max].clone().float()
        clip_image = F.pad(clip_image, (w_l, w_r, h_l, h_r, 0, 0))
    #clip_image = torch.nn.functional.interpolate(clip_image, scale_factor=(1, 2, 2), mode='trilinear').squeeze()
    
    clip_mask_backup = seg.clone()
    if crop_depth:
        clip_mask = clip_mask_backup[None, None, d_min:d_max, h_min:h_max, w_min:w_max]
        clip_mask = F.pad(clip_mask, (w_l, w_r, h_l, h_r, d_l, d_r))
    else:
        clip_mask = clip_mask_backup[None, None, :, h_min:h_max, w_min:w_max].clone().float()
        clip_mask = F.pad(clip_mask, (w_l, w_r, h_l, h_r, 0, 0))
    #clip_mask = torch.nn.functional.interpolate(clip_mask.float(), scale_factor=(1, 2, 2), mode='trilinear').squeeze().bool()
    crop_pos = {
        'w_l': w_l,
        'w_r': w_r,
        'h_l': h_l,
        'h_r': h_r,
        'd_l': d_l,
        'd_r': d_r,
        'd_min': d_min,
        'd_max': d_max,
        'h_min': h_min,
        'h_max': h_max,
        'w_min': w_min,
        'w_max': w_max,
    }
    return clip_image, clip_mask, d_min, d_l, crop_pos


def preprocess_img_seg(img, seg, predict, img_path = None, rotate_k = -1, crop_edge = 64):
    print("preprocess input shape", img.shape, seg.shape)
    
    predict = torch.rot90(predict, k=rotate_k, dims=(1, 2))
    seg = torch.rot90(seg, k=rotate_k, dims=(1, 2))
    
    print("before crop", img.shape, seg.shape)
    img = img[:, :, crop_edge:-crop_edge, crop_edge:-crop_edge]
    seg = seg[:, crop_edge:-crop_edge, crop_edge:-crop_edge]
    predict = predict[:, crop_edge:-crop_edge, crop_edge:-crop_edge]
    print("after crop", img.shape, seg.shape)
    
    img = img.unsqueeze(0)
    seg = seg.unsqueeze(0).unsqueeze(0)
    print("img mask shape:", img.shape, seg.shape)

    input_ct_img = img[0][0].clone()
    
    ct_image = torch.load(img_path)
    if ct_image.shape[-1] != 512:
        ct_image = ct_image.permute(2, 1, 0)
    ct_image = ct_image[:, crop_edge:-crop_edge, crop_edge:-crop_edge]
    print(ct_image.shape)
    
    ct_image = preprocess_ct_image(ct_image, img_path.split('/')[-1]) 
    assert ct_image.min() >= 0
    assert ct_image.max() <= 255

    ct_image = torch.rot90(ct_image, k=rotate_k, dims=(1, 2))
    input_ct_img = torch.rot90(input_ct_img, k=rotate_k, dims=(1, 2))
    
    return ct_image, input_ct_img, seg, predict


def run_software(val_data, test_idx, networks, args, logger):
    method = args.method
    device = args.device
    screen_rate = args.screen_size
    user_name = args.user_name
    logger.info(f'正在标注{test_idx}')
       
    img, seg, spacing, path = val_data[test_idx]
    img_name = path.split('/')[-1]
    
    if "LUNG" in img_name:
        crop_edge = 70
    elif "Dataset2" in img_name:
        crop_edge = 10
    elif "lung" in img_name:
        crop_edge = 50
    else:
        crop_edge = 1

    predict = torch.load("Test0407_predicts/"+img_name)
    
    final_result_mask = predict.float()
    
    if "lung" in img_name:
        rotate_k = -1
    else:
        rotate_k = -3
    print("rotate", img_name, rotate_k)
    ct_image, input_ct_img, seg, final_result_mask = preprocess_img_seg(img, seg, final_result_mask, img_path=path, rotate_k=rotate_k, crop_edge=crop_edge)
    ct_image_shape = ct_image.shape
    print("读取CT形状:", ct_image_shape)
    
    init_slice_idx = torch.argmax(final_result_mask.sum(-1).sum(-1))

    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    print(final_result_mask.shape, seg.shape)
    init_iou = dice_loss(final_result_mask.unsqueeze(0).unsqueeze(0), seg)
    logger.info(f'数据{test_idx}的初始Dice分数为: {1 - float(init_iou)}')

    overview_image_size = int(640 * screen_rate)
    zoom_in_size = overview_image_size
    interval_size = 10
    zoom_in_position = (overview_image_size + interval_size, 0)
    button_size = (200, 48)
    button_font_size = 30
    button_left_edege = 10
    
    window_width = overview_image_size * 2 + interval_size + button_size[0] + button_left_edege * 3
    window_height = overview_image_size + 100
    click_color_positive = (255, 0, 0)
    click_color_negative = (0, 0, 255)
    click_circle_radius = 3
    current_image_index = init_slice_idx # TODO change this
    logger.info(f'模型切换slice至{current_image_index}')

    h_interval = 60
    
    button_pos_x = overview_image_size * 2 + interval_size + button_left_edege
    #h_start = (button_pos_x - h_interval * 8 + h_interval - button_size[1]) / 2
    h_start = 20
    
    buttons = [
        Button((button_pos_x + button_left_edege, h_start), button_size, color=(239, 213, 149), text='[W] Last Slice', font_size=button_font_size),
        Button((button_pos_x + button_left_edege, h_start+h_interval*1), button_size, color=(239, 213, 149), text='[S] Next Slice', font_size=button_font_size),
        Button((button_pos_x + button_left_edege, h_start+h_interval*2), button_size, color=(255, 128, 128), text='[Z] Delete Click', font_size=button_font_size),
        Button((button_pos_x + button_left_edege, h_start+h_interval*3), button_size, color=(236, 238, 129), text='[G] Generate', font_size=button_font_size),
        Button((button_pos_x + button_left_edege, h_start+h_interval*4), button_size, color=(239, 180, 149), text='[H] Hide Mask', font_size=button_font_size),
        Button((button_pos_x + button_left_edege, h_start+h_interval*5), button_size, color=(239, 180, 149), text='[J] Show Mask', font_size=button_font_size),
        Button((button_pos_x + button_left_edege, h_start+h_interval*6), button_size, color=(210, 224, 251), text='[O] Save', font_size=button_font_size),
        Button((button_pos_x + button_left_edege, h_start+h_interval*7), button_size, color=(130, 160, 216), text='[P] Next CT', font_size=button_font_size)
    ]

    window = pygame.display.set_mode((window_width, window_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()

    show_mask = False

    click_positions = []
    global_click_positions = []

    background_color = (205, 250, 213)

    show_zoom_in_image = False
    
    def click_window_to_global(center_pos_global, window_clicks, clip_size=128):

        ret = []
        for this_click_window in window_clicks:
        
            window_center = torch.tensor([clip_size/2, clip_size/2])
            
            relative_pos = this_click_window - window_center
        
            ret.append(center_pos_global + relative_pos)
        
        return ret
        
        

    def draw_text(s, x, y, screen, color='black'):
        font = pygame.font.Font(None, 36)

        if color == 'black':
            text = font.render(s, True, (0, 0, 0))
        else:
            text = font.render(s, True, (1, 0, 0))

        text_rect = text.get_rect()

        text_rect.x = x
        text_rect.y = y

        screen.blit(text, text_rect)

    def make_surface_rgba(array):
        """Returns a surface made from a [w, h, 4] numpy array with per-pixel alpha
        """
        shape = array.shape
        if len(shape) != 3 and shape[2] != 4:
            raise ValueError("Array not RGBA")

        # Create a surface the same width and height as array and with
        # per-pixel alpha.
        surface = pygame.Surface(shape[0:2], pygame.SRCALPHA, 32)

        # Copy the rgb part of array to the new surface.
        pygame.pixelcopy.array_to_surface(surface, array[:,:,0:3])

        # Copy the alpha part of array to the surface using a pixels-alpha
        # view of the surface.
        surface_alpha = np.array(surface.get_view('A'), copy=False)
        surface_alpha[:,:] = array[:,:,3]

        return surface
    
    def process_clicks(click_positions):
        points_pos = []
        points_neg = []
        
        for position in click_positions:
            _image_index, _x, _y, click_type = position
            
            x = _x - zoom_in_position[0]
            y = _y - zoom_in_position[1]

            x = int(x / zoom_in_size * 128)
            y = int(y / zoom_in_size * 128)
            
            _image_index = _image_index - d_min
            
            if click_type == 1:
                points_pos.append([_image_index, x, y])
            elif click_type == 0:
                points_neg.append([_image_index, x, y])
            else:
                raise "Click label should be 0 or 1."
                
        points_pos = torch.tensor(points_pos)
        points_neg = torch.tensor(points_neg)
        return points_pos, points_neg
    
    software_start_time = time.time()
    
    running = True
    finish = False
    zoom_in_center_pos_global = torch.tensor([-10000, -10000])
    clip_image_for_infer = None
    
    while running:
        
        current_time = time.time()
        if (current_time - software_start_time) % 5 <= 0.1:
            # print("5 second pass")
            if not final_result_mask is None:
                # print("aoto save mask", final_result_mask.shape)
                torch.save(final_result_mask, f"exp_iseg_output/{user_name}/output_{img_name}")
                torch.save(global_click_positions, f"exp_iseg_output/{user_name}/click_positions_{img_name}")
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    if current_image_index < len(ct_image) - 1:
                        current_image_index = current_image_index + 1
                        logger.info(f'用户切换slice至{current_image_index}')
                elif event.key == pygame.K_w:
                    if current_image_index > 0:
                        current_image_index = current_image_index - 1
                        logger.info(f'用户切换slice至{current_image_index}')
                elif event.key == pygame.K_g:
                    logger.info(f'用户点击生成')
                    if method == 'tri_attn_loraAdapter_pEncodeS_miniDe' or method == 'baidu_XR':
                        points_pos, points_neg = process_clicks(click_positions)
                        predict_mask = model_inference(networks, clip_image_for_infer, clip_mask, points_pos, points_neg, device=device)
                    elif method == 'baidu_new':
                        points_pos, points_neg = process_clicks(click_positions)
                        predict_mask = model_inference_baidu(networks, clip_image_for_infer, clip_mask, points_pos, points_neg, device=device)
                    else:
                        raise "unknown method"

                    show_mask = True
                    patch_size = 128
                    predict_part = predict_mask.cpu()[crop_pos_dict['d_l']:patch_size-crop_pos_dict['d_r'], crop_pos_dict['h_l']:patch_size-crop_pos_dict['h_r'], crop_pos_dict['w_l']:patch_size-crop_pos_dict['w_r']]
                    print("current_image_index", current_image_index)
                    print("crop min max:", crop_pos_dict['d_min'], crop_pos_dict['d_max'])
                    edit_frames = []
                    affect_frame_range = 3
                    for edit_i in range(current_image_index - crop_pos_dict['d_min'] - affect_frame_range, current_image_index - crop_pos_dict['d_min'] + affect_frame_range + 1):
                        if edit_i >= 0:
                            edit_frames.append(edit_i)
                    print("edit frames", edit_frames)
                    if current_image_index - affect_frame_range >= 0:
                        final_result_mask[current_image_index - affect_frame_range: current_image_index + affect_frame_range + 1,
                                    crop_pos_dict['h_min']: crop_pos_dict['h_max'],
                                    crop_pos_dict['w_min']: crop_pos_dict['w_max']
                                    ] = predict_part[edit_frames, :, :]
                    else:
                        final_result_mask[0: current_image_index + affect_frame_range + 1,
                                    crop_pos_dict['h_min']: crop_pos_dict['h_max'],
                                    crop_pos_dict['w_min']: crop_pos_dict['w_max']
                                    ] = predict_part[edit_frames, :, :]
                    
                    now_iou = dice_loss(final_result_mask.unsqueeze(0).unsqueeze(0), seg)
                    logger.info(f'数据{test_idx}的当前全局Dice分数为: {1 - float(now_iou)}')
                    
                elif event.key == pygame.K_h:
                    if show_mask:
                        show_mask = False
                    else:
                        show_mask = True
                elif event.key == pygame.K_z:
                    if len(click_positions) > 0:
                        click_positions.pop(-1)
                        global_click_positions.pop(-1)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                if event.button == 1:  # 左键
                    # print("click on", mouse_pos)
                    logger.info(f'用户点击左键: {mouse_pos}')
                    # click_poses[current_image_index].append((mouse_pos, 1))
                    x, y = mouse_pos
                    if (zoom_in_position[0] < x < zoom_in_position[0] + zoom_in_size) and (zoom_in_position[1] < y < zoom_in_position[1] + zoom_in_size):                 
                        click_positions.append((current_image_index, x, y, 1))

                        points_pos, _ = process_clicks([(current_image_index, x, y, 1)])
                        global_pos = click_window_to_global(zoom_in_center_pos_global, points_pos[:, 1:])
                        g_x, g_y = global_pos[0]
                        global_click_positions.append((current_image_index, g_x, g_y, 1))
                        
                        logger.info(f'用户增加一个正向点击，目前点击数：{len(click_positions)}')
                    
                    for button_id, button in enumerate(buttons):
                        if button.is_clicked(mouse_pos):
                            # print(f"Button {button_id} Clicked!")
                            logger.info(f'用户点击按钮【{button_id}】')
                            if button_id == 0:
                                if current_image_index > 0:
                                    # current_image_index = (current_image_index - 1) % len(ct_image)
                                    current_image_index = current_image_index - 1
                                    logger.info(f'用户切换slice至{current_image_index}')
                            if button_id == 1:
                                if current_image_index < len(ct_image) - 1:
                                    current_image_index = current_image_index + 1
                                    logger.info(f'用户切换slice至{current_image_index}')
                            if button_id == 2:
                                if len(click_positions) > 0:
                                    
                                    click_positions.pop(-1)
                                    global_click_positions.pop(-1)
                                    logger.info(f'用户删除点击，目前剩余{len(click_positions)}点击')
                            if button_id == 3:
                                if not clip_image_for_infer is None:
                                    
                                    logger.info(f'用户点击生成')
                                    #logger.info(f'用户所有点击位置：',click_positions)
                                    if method == 'tri_attn_loraAdapter_pEncodeS_miniDe' or method == 'baidu_XR':
                                        points_pos, points_neg = process_clicks(click_positions)
                                        predict_mask = model_inference(networks, clip_image_for_infer, clip_mask, points_pos, points_neg, device=device)
                                    elif method == 'baidu_new':
                                        points_pos, points_neg = process_clicks(click_positions)
                                        predict_mask = model_inference_baidu(networks, clip_image_for_infer, clip_mask, points_pos, points_neg, device=device)
                                    else:
                                        raise "unknown method"
                                                
                                    show_mask = True
                                    patch_size = 128
                                    predict_part = predict_mask.cpu()[crop_pos_dict['d_l']:patch_size-crop_pos_dict['d_r'], crop_pos_dict['h_l']:patch_size-crop_pos_dict['h_r'], crop_pos_dict['w_l']:patch_size-crop_pos_dict['w_r']]
                                    print("current_image_index", current_image_index)
                                    print("crop min max:", crop_pos_dict['d_min'], crop_pos_dict['d_max'])
                                    edit_frames = []
                                    affect_frame_range = 3
                                    for edit_i in range(current_image_index - crop_pos_dict['d_min'] - affect_frame_range, current_image_index - crop_pos_dict['d_min'] + affect_frame_range + 1):
                                        if edit_i >= 0:
                                            edit_frames.append(edit_i)
                                    print("edit frames", edit_frames)
                                    if current_image_index - affect_frame_range >= 0:
                                        final_result_mask[current_image_index - affect_frame_range: current_image_index + affect_frame_range + 1,
                                                    crop_pos_dict['h_min']: crop_pos_dict['h_max'],
                                                    crop_pos_dict['w_min']: crop_pos_dict['w_max']
                                                    ] = predict_part[edit_frames, :, :]
                                    else:
                                        final_result_mask[0: current_image_index + affect_frame_range + 1,
                                                    crop_pos_dict['h_min']: crop_pos_dict['h_max'],
                                                    crop_pos_dict['w_min']: crop_pos_dict['w_max']
                                                    ] = predict_part[edit_frames, :, :]
                                    
                                    now_iou = dice_loss(final_result_mask.unsqueeze(0).unsqueeze(0), seg)
                                    logger.info(f'数据{test_idx}的当前全局Dice分数为: {1 - float(now_iou)}')
                                else:
                                    print("Please choose an area first.")
                                
                            if button_id == 4:
                                if show_mask:
                                    show_mask = False
                                else:
                                    show_mask = True
                            if button_id == 5:
                                show_mask = True
                            if button_id == 6:
                                print("save mask", final_result_mask.shape)
                                torch.save(final_result_mask, f"exp_iseg_output/output_{img_name}")
                                torch.save(global_click_positions, f"exp_iseg_output/click_positions_{img_name}")
                            if button_id == 7:
                                logger.info(f'完成标注，进入下一个CT')
                                test_idx = test_idx + 1
                                
                                torch.save(final_result_mask, f"exp_iseg_output/{user_name}/output_{img_name}")
                                torch.save(global_click_positions, f"exp_iseg_output/{user_name}/click_positions_{img_name}")
                                
                                
                                file_path = "utils/all_users.txt"
                                previous_user_names = []

                                ids_to_names = ["Default" for i in range(20)]
                                names_to_id = {}
                                names_to_target = {}
                                
                                with open(file_path, 'r') as file:
                                    for line in file:
                                        info = line.rstrip('\n')
                                        if "ID" not in info:
                                            continue
                                        name = info.split(' - ID:')[0]
                                        id = info.split(' - ID:')[1].split(' - CurrentTargetIdx:')[0]
                                        current_target_idx = info.split(' - CurrentTargetIdx:')[1]
                                        id = int(id)
                                        previous_user_names.append(name)
                                        names_to_id[name] = id
                                        names_to_target[name] = current_target_idx

                                names_to_target[args.user_name] = test_idx
                                
                                print("names_to_target", names_to_target)
                                with open(file_path, 'w') as file:
                                    for name in previous_user_names:
                                        file.writelines(name+" - ID:"+str(names_to_id[name]) + " - CurrentTargetIdx:" + str(names_to_target[name]) + "\n")
                                
                                if test_idx < len(val_data):
                                    img, seg, spacing, path = val_data[test_idx]
                                    img_name = path.split('/')[-1]
                                    
                                    if "LUNG" in img_name:
                                        crop_edge = 70
                                    elif "Dataset2" in img_name:
                                        crop_edge = 10
                                    elif "lung" in img_name:
                                        crop_edge = 50
                                    else:
                                        crop_edge = 1
                                
                                    logger.info(f'正在标注{test_idx}')
                                    
                                    #predict = torch.load("Test0328_predicts/"+img_name)
                                    predict = torch.load("Test0407_predicts/"+img_name)
                                    
                                    
                                    if predict.shape[-1] != 512:
                                        predict = predict.permute(2, 1, 0)
                                    assert predict.shape[-1] == 512
                                    
                                    final_result_mask = predict.float()
                                    print("load final_result_mask shape", final_result_mask.shape)
                                    if "lung" in img_name:
                                        rotate_k = -1
                                    else:
                                        rotate_k = -3
                                    print(img_name, rotate_k)
                                    ct_image, input_ct_img, seg, final_result_mask = preprocess_img_seg(img, seg, final_result_mask, img_path=path, rotate_k=rotate_k, crop_edge=crop_edge)
                                    ct_image_shape = ct_image.shape
                                    print("读取CT形状:", img_name, ct_image_shape)
                                    show_zoom_in_image = False
                                    show_mask = False
                                    click_positions = []
                                    global_click_positions = []
                                    
                                    init_slice_idx = torch.argmax(final_result_mask.sum(-1).sum(-1))
                                    init_iou = dice_loss(final_result_mask.unsqueeze(0).unsqueeze(0), seg)
                                    logger.info(f'数据{test_idx}的初始Dice分数为: {1 - float(init_iou)}')
                                    current_image_index = init_slice_idx
                                    logger.info(f'模型切换slice至{current_image_index}')
                                else:
                                    logger.info(f'所有标注已完成')
                                    finish = True
                                    
                            
                elif event.button == 2:  # 中键
                    logger.info(f'用户点击中键进行裁剪：{mouse_pos}')
                    x, y = mouse_pos
                        
                    original_size = 512 - crop_edge * 2
                    x = int(x * original_size / overview_image_size)
                    y = int(y * original_size / overview_image_size)
                        
                    z = current_image_index
                    
                    zoom_in_center_slice_index = current_image_index
                    zoom_in_center_pos_global = torch.tensor([x, y])

                    clip_image_for_infer, clip_mask, d_min, d_l, crop_pos_dict = crop_by_click_position(input_ct_img, seg, center_pos=[x, y, z])  # TODO: maybe this is x,y,z
                    
                    clip_image, predict_mask, d_min, _, _ = crop_by_click_position(ct_image, final_result_mask.unsqueeze(0).unsqueeze(0), center_pos=[x, y, z], crop_depth=False)
                    
                    print("裁剪", d_min, d_l)
                    clip_image = clip_image.squeeze()
                    predict_mask = predict_mask.squeeze()
                    
                    show_zoom_in_image = True
                    show_mask = True
                    click_positions = []
                    
                elif event.button == 3:  # 右键
                    logger.info(f'用户点击右键：{mouse_pos}')
                    x, y = mouse_pos
                    if (zoom_in_position[0] < x < zoom_in_position[0] + zoom_in_size) and (zoom_in_position[1] < y < zoom_in_position[1] + zoom_in_size):
                        click_positions.append((current_image_index, x, y, 0))
                        logger.info(f'用户增加一个负向点击，目前点击数：{len(click_positions)}')
                        
                        _, points_neg = process_clicks([(current_image_index, x, y, 0)])
                        points_neg = click_window_to_global(zoom_in_center_pos_global, points_neg[:, 1:])
                        g_x, g_y = points_neg[0]
                        global_click_positions.append((current_image_index, g_x, g_y, 0))
                        
                elif event.button == 4:  # 上滚
                    if current_image_index > 0:
                        current_image_index = current_image_index - 1
                        logger.info(f'用户切换slice至{current_image_index}')
                    # current_image_index = (current_image_index - 1) % len(ct_image)
                elif event.button == 5:  # 下滚
                    if current_image_index < len(ct_image) - 1:
                        current_image_index = current_image_index + 1
                        logger.info(f'用户切换slice至{current_image_index}')
                    # current_image_index = (current_image_index + 1) % len(ct_image)
                    
        window.fill(background_color)
        current_image = ct_image[current_image_index].float()
        current_image = F.interpolate(current_image[None, None, :, :], size=(overview_image_size, overview_image_size),mode='bilinear').squeeze()
        current_image_3d = torch.cat([
            current_image.unsqueeze(-1), 
            current_image.unsqueeze(-1), 
            current_image.unsqueeze(-1)], -1)
        current_image = pygame.surfarray.make_surface(current_image_3d.int().numpy())
        window.blit(current_image, (0, 0))

        zoom_in_this_frame_mask = final_result_mask[current_image_index].detach().cpu()
        
        zoom_in_this_frame_mask = F.interpolate(zoom_in_this_frame_mask[None, None, :, :], size=(zoom_in_size, zoom_in_size),mode='bilinear').squeeze()
        
        zoom_in_this_frame_mask_3d = torch.cat([
            torch.zeros_like(zoom_in_this_frame_mask.unsqueeze(-1)), 
            zoom_in_this_frame_mask.unsqueeze(-1) * 128, 
            torch.zeros_like(zoom_in_this_frame_mask.unsqueeze(-1)),
            zoom_in_this_frame_mask.unsqueeze(-1) * 128
            ], -1)
        
        zoom_in_this_frame_mask_3d_vis = make_surface_rgba(zoom_in_this_frame_mask_3d.int().numpy())
        window.blit(zoom_in_this_frame_mask_3d_vis, (0, 0))
        
        if show_zoom_in_image:
            current_is_zoom_in_range = zoom_in_center_slice_index - 64 < current_image_index < zoom_in_center_slice_index + 64
            show_zoom_in_image = show_zoom_in_image and current_is_zoom_in_range
        
        if show_zoom_in_image:
            
            clip_image_vis = clip_image[current_image_index]

            clip_image_vis = F.interpolate(clip_image_vis[None, None, :, :], size=(zoom_in_size, zoom_in_size),mode='bilinear').squeeze()
            clip_image_3d = torch.cat([
                clip_image_vis.unsqueeze(-1), 
                clip_image_vis.unsqueeze(-1), 
                clip_image_vis.unsqueeze(-1)], -1)
            zoom_in_image = pygame.surfarray.make_surface(clip_image_3d.int().numpy())
            window.blit(zoom_in_image, zoom_in_position)
            
            if show_mask:
                
                zoom_in_this_frame_mask = final_result_mask[current_image_index,
                                                  crop_pos_dict['h_min']: crop_pos_dict['h_max'],
                                                  crop_pos_dict['w_min']: crop_pos_dict['w_max']
                                                  ].detach().cpu()
                zoom_in_this_frame_mask = F.pad(zoom_in_this_frame_mask[None, None, :, :],
                                                (crop_pos_dict['w_l'], crop_pos_dict['w_r'], crop_pos_dict['h_l'], crop_pos_dict['h_r'])
                                                ).squeeze()
                try:
                    zoom_in_this_frame_mask = F.interpolate(zoom_in_this_frame_mask[None, None, :, :], size=(zoom_in_size, zoom_in_size),mode='bilinear').squeeze()
                except:
                    print("Error ------------------------------------------------")
                    print(crop_pos_dict)
                    print(final_result_mask.shape)
                
                zoom_in_this_frame_mask_3d = torch.cat([
                    torch.zeros_like(zoom_in_this_frame_mask.unsqueeze(-1)), 
                    zoom_in_this_frame_mask.unsqueeze(-1) * 128, 
                    torch.zeros_like(zoom_in_this_frame_mask.unsqueeze(-1)),
                    zoom_in_this_frame_mask.unsqueeze(-1) * 128
                    ], -1)
                
                zoom_in_this_frame_mask_3d_vis = make_surface_rgba(zoom_in_this_frame_mask_3d.int().numpy())
                window.blit(zoom_in_this_frame_mask_3d_vis, zoom_in_position)
            
        for button in buttons:
            button.draw(window)
        draw_text(f"Current Slice: {current_image_index}", 0, overview_image_size + 15, window)
        draw_text(f"Current CT: {str(test_idx)}", 0, overview_image_size + 50, window)
        
        if not show_zoom_in_image:
            draw_text("Click left to choose a area please.", overview_image_size + 50, overview_image_size // 2, window)
        if finish:
            draw_text(f"Annotation process completed. ", 400, overview_image_size + 15, window, color='red')
            draw_text(f"Thank you for your contribution!", 400, overview_image_size + 50, window, color='red')

        for pos in click_positions:
            _image_index, x, y, click_type = pos
            if _image_index != current_image_index:
                continue
            if click_type == 1:
                pygame.draw.circle(window, click_color_positive, (x, y), click_circle_radius)
            else:
                pygame.draw.circle(window, click_color_negative, (x, y), click_circle_radius)
                
        for pos in global_click_positions:
            _image_index, x, y, click_type = pos
            screen_g_x = int(x / original_size * overview_image_size)
            screen_g_y = int(y / original_size * overview_image_size)
            if _image_index != current_image_index:
                continue
            if click_type == 1:
                pygame.draw.circle(window, click_color_positive, (screen_g_x, screen_g_y), click_circle_radius)
            else:
                pygame.draw.circle(window, click_color_negative, (screen_g_x, screen_g_y), click_circle_radius)

        pygame.display.flip() 

        clock.tick(120) 
    pygame.quit()

def get_random_clicks(_seg, num = 5, is_positive = True):
    if len(_seg.shape) == 5:
        _seg = _seg.squeeze(1)
    if len(_seg.shape) == 3:
        _seg = _seg.unsqueeze(0)
    assert len(_seg.shape) == 4
    if is_positive:
        l = len(torch.where(_seg == 1)[0])
        np.random.seed(20241)
        sample = np.random.choice(np.arange(l), num, replace=True)
        x = torch.where(_seg == 1)[1][sample].unsqueeze(1)
        y = torch.where(_seg == 1)[2][sample].unsqueeze(1)
        z = torch.where(_seg == 1)[3][sample].unsqueeze(1)
        points_pos = torch.cat([x, y, z], dim=1).float()#.to(device)
        return points_pos
    else:
        l = len(torch.where(_seg == 0)[0])
        np.random.seed(20242)
        sample = np.random.choice(np.arange(l), num, replace=True)
        x = torch.where(_seg == 0)[1][sample].unsqueeze(1)
        y = torch.where(_seg == 0)[2][sample].unsqueeze(1)
        z = torch.where(_seg == 0)[3][sample].unsqueeze(1)
        points_neg = torch.cat([x, y, z], dim=1).float()#.to(device)
        return points_neg


def test_networks(networks, img, seg):
    if len(img.shape) == 4:
        img = img.unsqueeze(0)
    if len(img.shape) == 3:
        img = img.unsqueeze(0).unsqueeze(0)
    if len(seg.shape) == 3:
        seg = seg.unsqueeze(0).unsqueeze(0)
    if len(seg.shape) == 4:
        seg = seg.unsqueeze(0)
    
    assert len(img.shape) == 5
    if img.shape[1] != 3:
        img = img.repeat(1, 3, 1, 1, 1)

    img_encoder, prompt_encoder, mask_decoder = networks
    device = 'cuda'
    input_image_size = 256
    patch_size = 128
    
    dis_map = DistMaps(2, use_disks=True)
    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    
    # img = img.unsqueeze(0)
    # seg = seg.unsqueeze(0)
    
    out = F.interpolate(img.float(), scale_factor=input_image_size / patch_size, mode='trilinear')  # TODO: for 256
    input_batch = out.to(device)
    batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
    input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)
    
    seg = seg.to(device)
    
    # randomly sample click points
    this_batch_points_feature = []

    assert seg.shape[-1] == 128
    
    l = len(torch.where(seg == 1)[0])
    if l > 0:
        points_pos = get_random_clicks(seg, num = 5, is_positive = True)
        #print("random points_pos", points_pos)
        positive_feat = dis_map.get_coord_features(points_pos, 1, 128, 128, 128)
    else:
        print("no target")
        positive_feat = torch.zeros(1, 1, 128, 128, 128).to(device)
        raise "why not target"

    points_neg = get_random_clicks(seg, num = 1, is_positive = False)
    #print("random points_neg", points_neg)
    negative_feat = dis_map.get_coord_features(points_neg, 1, 128, 128, 128)
    
    this_batch_points_feature.append(
        torch.cat([positive_feat, negative_feat], dim=1)
    )

    prompt_input = torch.cat(this_batch_points_feature, 0).float()
    
    point_feature = prompt_encoder(prompt_input)
    batch_features = img_encoder(input_batch, batchsize, point_feature)  # bs, C, H, W, D
    
    masks = mask_decoder(batch_features)
    masks = masks.permute(0, 1, 4, 2, 3)  # bs, C, D, H, W

    loss = dice_loss(masks, seg)
    
    print("test loss", 1 - float(loss))
    
    masks = F.softmax(masks, dim=1)[:,1][0]
    masks = masks.squeeze()

    return masks


def model_inference(networks, img, seg, points_pos, points_neg, device):
    print("model inference ", device)
    with torch.no_grad():
        if len(img.shape) == 4:
            img = img.unsqueeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0).unsqueeze(0)
        if len(seg.shape) == 3:
            seg = seg.unsqueeze(0).unsqueeze(0)
        if len(seg.shape) == 4:
            seg = seg.unsqueeze(0)
        
        assert len(img.shape) == 5
        if img.shape[1] != 3:
            img = img.repeat(1, 3, 1, 1, 1)

        img_encoder, prompt_encoder, mask_decoder = networks

        input_image_size = 256
        patch_size = 128
        
        dis_map = DistMaps(2, use_disks=True)
        dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
        
        out = F.interpolate(img.float(), scale_factor=input_image_size / patch_size, mode='trilinear')  # TODO: for 256
        input_batch = out.to(device)
        batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
        input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)
        
        seg = seg.to(device)
        points_pos = points_pos.to(device)
        points_neg = points_neg.to(device)
        
        # randomly sample click points
        this_batch_points_feature = []

        assert seg.shape[-1] == 128
        
        l = len(torch.where(seg == 1)[0])
        if len(points_pos) > 0:
            # points_pos = get_random_clicks(seg, num = 5, is_positive = True)
            print("random points_pos", points_pos)
            positive_feat = dis_map.get_coord_features(points_pos, 1, 128, 128, 128)
        else:
            print("no target")
            positive_feat = torch.zeros(1, 1, 128, 128, 128).to(device)
            # raise "why not target"
            return torch.zeros(128, 128, 128)

        # points_neg = get_random_clicks(seg, num = 1, is_positive = False)
        print("random points_neg", points_neg)
        if len(points_neg) > 0:
            negative_feat = dis_map.get_coord_features(points_neg, 1, 128, 128, 128)
        else:
            negative_feat = torch.zeros(1, 1, 128, 128, 128).to(device)
        
        this_batch_points_feature.append(
            torch.cat([positive_feat, negative_feat], dim=1)
        )

        prompt_input = torch.cat(this_batch_points_feature, 0).float()
        
        point_feature = prompt_encoder(prompt_input)
        batch_features = img_encoder(input_batch, batchsize, point_feature)  # bs, C, H, W, D
        
        masks = mask_decoder(batch_features)
        masks = masks.permute(0, 1, 4, 2, 3)  # bs, C, D, H, W

        loss = dice_loss(masks, seg)
        
        print("test loss", float(loss))
        
        masks = F.softmax(masks, dim=1)[:,1][0]
        masks = masks.squeeze()
        masks = masks > 0.5
        masks = masks.float()
        print("预测最大概率", masks.max())

    return masks

def model_inference_baidu(networks, img, seg, points_pos, points_neg, device):
    with torch.no_grad():
        if len(img.shape) == 4:
            img = img.unsqueeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0).unsqueeze(0)
        if len(seg.shape) == 3:
            seg = seg.unsqueeze(0).unsqueeze(0)
        if len(seg.shape) == 4:
            seg = seg.unsqueeze(0)
        
        assert len(img.shape) == 5
        if img.shape[1] != 3:
            img = img.repeat(1, 3, 1, 1, 1)

        img_encoder, prompt_encoder, mask_decoder = networks

        input_image_size = 256
        patch_size = 128
        
        dis_map = DistMaps(2, use_disks=True)
        dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
        
        out = F.interpolate(img.float(), scale_factor=input_image_size / patch_size, mode='trilinear')  # TODO: for 256
        input_batch = out.to(device)
        batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
        input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)
        
        seg = seg.to(device)
        points_pos = points_pos.to(device)
        points_neg = points_neg.to(device)
        
        # randomly sample click points
        this_batch_points_feature = []

        assert seg.shape[-1] == 128
        
        print("points", points_pos.shape, points_neg.shape)
        points_torch = torch.cat([points_pos, points_neg], dim = 0)
        
        if len(points_torch) != 0:
            points_torch = points_torch.unsqueeze(0)  # bs, num_clicks, 3
            labels_ones = torch.ones(len(points_pos))
            labels_zeros = torch.zeros(len(points_neg))
            labels_torch = torch.cat([labels_ones, labels_zeros], 0).unsqueeze(0)  # bs, 10
            prompt_input = [points_torch, labels_torch]
            point_feature, _dense = prompt_encoder(prompt_input, None, None)
        else:
            point_feature = None
        
        batch_features = img_encoder(input_batch, batchsize)
        image_pe = prompt_encoder.get_dense_pe().to(device)
        
        masks = mask_decoder(batch_features, image_pe, point_feature, None, True)
        masks = masks.permute(0, 1, 4, 2, 3)  # bs, C, D, H, W

        loss = dice_loss(masks, seg)
        
        print("test loss", float(loss))
        logger.info(f'数据{test_idx}的当前Dice分数为:', float(loss))
        
        masks = F.softmax(masks, dim=1)[:,1][0]
        masks = masks > 0.5
        masks = masks.squeeze()

    return masks

if __name__ == "__main__":
    
    
    val_data, networks, args, logger = build_network_and_data()
    
    test_idx = int(args.current_target_idx)
    #test_idx = 0
    
    print("args device", args.device)
    run_software(val_data, test_idx, networks, args, logger)