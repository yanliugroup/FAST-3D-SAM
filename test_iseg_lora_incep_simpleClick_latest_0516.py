from dataset.datasets import load_data_volume
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
from torch.optim import AdamW
import numpy as np
import logging
from utils.script_util import save_checkpoint
import sys
from monai.losses import DiceCELoss, DiceLoss
import ast
import torch.nn.functional as F
from modeling.Med_SAM.mask_decoder_mamba import MaskDecoder
import torch
from modeling.Med_SAM.prompt_encoder_simple import PromptEncoderS
from utils.click_encoding import DistMaps
import torch.nn as nn
from functools import partial
import os
from utils.util import setup_logger, count_parameters
import time
import matplotlib.pyplot as plt
import random
from monai.metrics import HausdorffDistanceMetric

def adjust_window(image, window_width = 1400, window_level = 400):
    window_min = window_level - window_width / 2
    window_max = window_level + window_width / 2

    adjusted_image = np.clip(image, window_min, window_max)

    adjusted_image = (adjusted_image - window_min) / (window_max - window_min)
    
    return adjusted_image

def visualize_CT_mask(ct, mask, predict, name):

    ct = adjust_window(ct)
    ct = ct * 255

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    axs[0].imshow(ct, cmap='gray')

    axs[1].imshow(ct, cmap='gray')

    axs[1].imshow(mask, alpha=0.5)

    axs[2].imshow(ct, cmap='gray')
    axs[2].imshow(predict, alpha=0.5)

    axs[0].set_title('CT')
    axs[1].set_title('CT & Mask')
    plt.title(name)

    plt.tight_layout()

    plt.savefig(f"exp_images/exp_images_simu_0407/{name}.png", dpi=120)
    plt.clf()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", default=None, type=str, choices=["cuhk", "baidu", "lora_incep", "tri_attn", "tri_attn_loraAdapter_pEncodeS_miniDe"]
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
        default="",
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
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--eval_interval", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_worker", default=12, type=int)
    parser.add_argument("-tolerance", default=5, type=int)

    args = parser.parse_args()

    if args.method == "cuhk":
        from modeling.Med_SAM.image_encoder_sam import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
    elif args.method == "baidu":
        from modeling.Med_SAM.image_encoder_baidu import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
    elif args.method == "lora_incep":
        from modeling.Med_SAM.image_encoder_lora_incep_sam import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
        import loralib as lora
    elif args.method == "tri_attn":
        from modeling.Med_SAM.image_encoder_tri_attn import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
        import loralib as lora
    elif args.method == "tri_attn_loraOnly":
        from modeling.Med_SAM.image_encoder_tri_attn_Lora import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
        import loralib as lora
    elif args.method == "tri_attn_loraAdapter_pEncodeS_miniDe":
        from modeling.Med_SAM.image_encoder_tri_attn_Lora_Adapter import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
        import loralib as lora
    else:
        raise "unknown method"
    input_image_size = args.input_image_size
    device = args.device
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
            
    args.weight_path = os.path.join(args.snapshot_path, "['lung','lung2','Lung421','pancreas','kits23','hepatic']")
    
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    setup_logger(logger_name="test", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"test")
    logger.info(str(args))

    args.data = ast.literal_eval(args.data)
    args.data_prefix = [f"../datafile/{dataset_name}_crop" for dataset_name in args.data]
    print(args.data_prefix)
    
    val_data = load_data_volume(
        datas=args.data,
        path_prefixes=args.data_prefix,
        batch_size=1,
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

    if args.method == "tri_attn_loraAdapter_pEncodeS_miniDe":
        for i in img_encoder.blocks:
            i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)

    
    load_pretrained = True
    file = "best_debug.pth.tar"
    print("load_pretrained", load_pretrained)
    if not load_pretrained:
        img_encoder.load_state_dict(mask_generator.predictor.model.image_encoder.state_dict(), strict=False)
    else:
        img_encoder.load_state_dict(torch.load(os.path.join(args.weight_path, file), map_location='cpu')["encoder_dict"], strict=True)  # TODO: change to True
    del sam
    img_encoder.to(device)

    prompt_encoder = PromptEncoderS(32)
    if load_pretrained:
        prompt_encoder.load_state_dict(
            torch.load(os.path.join(args.weight_path, file), map_location='cpu')["feature_dict"], strict=True)
    prompt_encoder.to(device)
    mask_decoder = MaskDecoder()
    
    if load_pretrained:
        mask_decoder.load_state_dict(torch.load(os.path.join(args.weight_path, file), map_location='cpu')["decoder_dict"],
                            strict=True)
    mask_decoder.to(device)

    logger.info(f'The img_encoder model has {count_parameters(img_encoder):,}M trainable parameters.')
    logger.info(f'The mask_decoder model has {count_parameters(mask_decoder):,}M trainable parameters.')

    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean", get_not_nans=True)

    patch_size = args.rand_crop_size[0]
    
    save_pred_path = os.path.join(args.snapshot_path, "predictions")
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)
    
    hd95_overview = []
    loss_overview = []
    mean_overview = []
    
    dis_map = DistMaps(2, use_disks=True)
    
    img_encoder.eval()
    prompt_encoder.eval()
    mask_decoder.eval()
    
    from thop import profile
    input_batch = torch.randn((256, 3, 256, 256)).to('cuda')
    input_batch_prompt = torch.randn((1, 768, 16, 16, 16)).to('cuda')
    macs_img_encoder, _ = profile(img_encoder, inputs=(input_batch, 1, input_batch_prompt))
    print("img_encoder MAC数量(G):", macs_img_encoder/1e9)
    
    input_batch = torch.randn((1, 2, 128, 128, 128)).to('cuda')
    macs_prompt_encoder, _ = profile(prompt_encoder, inputs=(input_batch,))
    print("macs_prompt_encoder MAC数量(G):", macs_prompt_encoder/1e9)
    
    input_batch_decode = torch.randn((1, 256, 16, 16, 16)).to('cuda')
    macs_mask_decoder, _ = profile(mask_decoder, inputs=(input_batch_decode, ))
    print("mask_decoder MAC数量(G):", macs_mask_decoder/1e9)
    
    print("模型总MAC数量（G）", (macs_img_encoder+macs_prompt_encoder+macs_mask_decoder)/1e9)
    
    total_infer_time = 0
    infer_num = 0
    
    click_num = 10
    
    with torch.no_grad():
        loss_summary = []
        #for idx, (img, seg, spacing) in enumerate(val_data):
        for idx, (img, seg, spacing, path) in enumerate(val_data):
            start_time = time.time()
            original_shape = img.shape
            original_img = img.clone()
            original_seg = seg.clone()
            
            img = F.interpolate(img.float(), size=128, mode='trilinear')
            seg = F.interpolate(seg.float().unsqueeze(1), size=128, mode='trilinear').squeeze(1)
            
            #print("resize shape", img.shape, seg.shape)
            
            out = F.interpolate(img.float(), scale_factor=input_image_size / patch_size, mode='trilinear')  # TODO: for 256
            input_batch = out.to(device)
            batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
            input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)
            
            seg = seg.to(device)
            
            # randomly sample click points
            this_batch_points_feature = []
            for _seg in seg:
                _seg = _seg.unsqueeze(0)
                assert _seg.shape[-1] == 128
                
                l = len(torch.where(_seg == 1)[0])
                if l > 0:
                    np.random.seed(20241)
                    sample = np.random.choice(np.arange(l), click_num, replace=True)
                    x = torch.where(_seg == 1)[1][sample].unsqueeze(1)
                    y = torch.where(_seg == 1)[2][sample].unsqueeze(1)
                    z = torch.where(_seg == 1)[3][sample].unsqueeze(1)
                    points_pos = torch.cat([x, y, z], dim=1).float()#.to(device)
                    positive_feat = dis_map.get_coord_features(points_pos, 1, 128, 128, 128)
                else:
                    print("no target")
                    positive_feat = torch.zeros(1, 1, 128, 128, 128).to(device)

                l = len(torch.where(_seg == 0)[0])
                np.random.seed(20242)
                sample = np.random.choice(np.arange(l), click_num, replace=True)
                x = torch.where(_seg == 0)[1][sample].unsqueeze(1)
                y = torch.where(_seg == 0)[2][sample].unsqueeze(1)
                z = torch.where(_seg == 0)[3][sample].unsqueeze(1)
                points_neg = torch.cat([x, y, z], dim=1).float()#.to(device)
                negative_feat = dis_map.get_coord_features(points_neg, 1, 128, 128, 128)
                
                this_batch_points_feature.append(
                    torch.cat([positive_feat, negative_feat], dim=1)
                )

            prompt_input = torch.cat(this_batch_points_feature, 0).float()
            
            point_feature = prompt_encoder(prompt_input)
            batch_features = img_encoder(input_batch, batchsize, point_feature)  # bs, C, H, W, D
            
            masks = mask_decoder(batch_features)
            masks = masks.permute(0, 1, 4, 2, 3)  # bs, C, D, H, W
            
            end_time = time.time()
            total_infer_time += end_time - start_time
            infer_num += len(masks)
            
            seg = seg.unsqueeze(1)
            loss = dice_loss(masks, seg)
            loss_summary.append(loss.detach().cpu().numpy())
            #print(path)
            
            masks = F.softmax(masks, dim=1)#[:, 1]
            masks = masks > 0.5
            hd95 = hd95_metric(masks, seg)
            

            
            loss_overview.append(loss.mean().detach().cpu().numpy())
            mean_overview.append(img.mean().detach().cpu().numpy().squeeze())
            if not torch.isnan(hd95):
                hd95_overview.append(hd95.mean().detach().cpu().numpy())
            
            
            logger.info(
                'iter: {}/{} '.format(idx, len(val_data))  + " : loss:" + str(
                    loss_summary[-1].flatten()[0]))

    print("平均推理时间", round(total_infer_time / infer_num, 4)) 
    logger.info("- Val metrics: " + str(np.mean(loss_summary)))
    print("dice score")
    print((1 - sum(loss_overview) / len(loss_overview)))
    print("hd95")
    print(sum(hd95_overview) / len(hd95_overview))


if __name__ == "__main__":
    main()

