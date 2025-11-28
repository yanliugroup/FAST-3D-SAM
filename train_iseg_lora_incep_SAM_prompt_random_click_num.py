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
from modeling.Med_SAM.mask_decoder_sam import MaskDecoder
import torch
from modeling.Med_SAM.prompt_encoder_sam import PromptEncoder, TwoWayTransformer
import torch.nn as nn
from functools import partial
import os
from utils.util import setup_logger, count_parameters
import time
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", default=None, type=str, choices=["sam", "baidu_new", "lora_incep", "tri_attn_loraOnly"]
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
    parser.add_argument("--max_clicks", default=12, type=int)
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--eval_interval", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_worker", default=12, type=int)
    parser.add_argument("-tolerance", default=5, type=int)

    args = parser.parse_args()

    if args.method == "sam":
        from modeling.Med_SAM.image_encoder_sam import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
    elif args.method == "baidu_new":
        from modeling.Med_SAM.image_encoder_baidu_new import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
    elif args.method == "lora_incep":
        from modeling.Med_SAM.image_encoder_lora_incep_sam import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
        import loralib as lora
    elif args.method == "tri_attn_loraOnly":
        from modeling.Med_SAM.image_encoder_tri_attn_Lora_woPrompt import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
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

    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))

    args.data = ast.literal_eval(args.data)
    args.data_prefix = [f"../datafile/{dataset_name}_crop" for dataset_name in args.data]
    print(args.data_prefix)

    train_data = load_data_volume(
        datas=args.data,
        path_prefixes=args.data_prefix,
        batch_size=args.batch_size,
        augmentation=True,
        split="train",
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker = args.num_worker
    )
    val_data = load_data_volume(
        datas=args.data,
        path_prefixes=args.data_prefix,
        batch_size=args.batch_size,
        augmentation=False,
        split="val",
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

    
    

    # Choose training weights
    if args.method == "baidu_new":
        for p in img_encoder.parameters():
            p.requires_grad = False
        img_encoder.depth_embed.requires_grad = True
        for p in img_encoder.slice_embed.parameters():
            p.requires_grad = True
        for i in img_encoder.blocks:
            for p in i.norm1.parameters():
                p.requires_grad = True
            for p in i.MLP_Adapter.parameters():
                p.requires_grad = True
            for p in i.Space_Adapter.parameters():
                p.requires_grad = True
            for p in i.Depth_Adapter.parameters():
                p.requires_grad = True
            for p in i.norm2.parameters():
                p.requires_grad = True
            i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)
        for p in img_encoder.neck.parameters():
            p.requires_grad = True
    elif args.method == "sam":
        for p in img_encoder.parameters():
            p.requires_grad = False
        img_encoder.depth_embed.requires_grad = True
        for p in img_encoder.slice_embed.parameters():
            p.requires_grad = True
        for i in img_encoder.blocks:
            for p in i.norm1.parameters():
                p.requires_grad = True
            # for p in i.adapter.parameters():
            #     p.requires_grad = True
            for p in i.norm2.parameters():
                p.requires_grad = True
            i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)
        for p in img_encoder.neck_3d.parameters():
            p.requires_grad = True
    
    elif args.method == "tri_attn_loraOnly":
        
        lora.mark_only_lora_as_trainable(img_encoder)
        logger.info(f'LORA The img_encoder model has {count_parameters(img_encoder):,}M trainable parameters.')
        
        img_encoder.depth_embed.requires_grad = True
        for p in img_encoder.slice_embed.parameters():
            p.requires_grad = True
        for i in img_encoder.blocks:
            for p in i.norm1.parameters():
                p.requires_grad = True
            for p in i.norm2.parameters():
                p.requires_grad = True
            i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)
        for p in img_encoder.neck.parameters():
            p.requires_grad = True
    
    elif args.method == "lora_incep":
        # for p in img_encoder.parameters():
        #     p.requires_grad = False
        lora.mark_only_lora_as_trainable(img_encoder)

        logger.info(f'LORA The img_encoder model has {count_parameters(img_encoder):,}M trainable parameters.')

        img_encoder.depth_embed.requires_grad = True
        for p in img_encoder.slice_embed.parameters():
            p.requires_grad = True
        for i in img_encoder.blocks:
            for p in i.norm1.parameters():
                p.requires_grad = True
            for p in i.adapter.parameters():
                p.requires_grad = True
            for p in i.norm2.parameters():
                p.requires_grad = True
            i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)
        for p in img_encoder.neck_3d.parameters():
            p.requires_grad = True

    
    
    load_pretrained = args.pretrained
    file = "best_debug.pth.tar"
    print("load_pretrained", load_pretrained)
    if not load_pretrained:
        img_encoder.load_state_dict(mask_generator.predictor.model.image_encoder.state_dict(), strict=False)
    else:
        img_encoder.load_state_dict(torch.load(os.path.join(args.weight_path, file), map_location='cpu')["encoder_dict"], strict=True)
    del sam
    img_encoder.to(device)
    
    prompt_encoder = PromptEncoder(256, [16, 16, 16], [128, 128, 128], mask_in_chans=16)
    if load_pretrained:
        prompt_encoder.load_state_dict(
            torch.load(os.path.join(args.weight_path, file), map_location='cpu')["feature_dict"], strict=True)
    prompt_encoder.to(device)
    
    """
    
    mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
    """

    
    two_way = TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048)
    two_way.load_state_dict(mask_generator.predictor.model.mask_decoder.transformer.state_dict(), strict=True)
    
    # weight_dict = two_way.state_dict()
    # for w_name in weight_dict.keys():
    #     print(w_name, weight_dict[w_name].shape)
    # raise "check weight"
    
    mask_decoder = MaskDecoder(transformer_dim=256, transformer=two_way, num_multimask_outputs=1)
    
    if load_pretrained:
        mask_decoder.load_state_dict(torch.load(os.path.join(args.weight_path, file), map_location='cpu')["decoder_dict"],
                            strict=True)
    mask_decoder.to(device)

    logger.info(f'The img_encoder model has {count_parameters(img_encoder):,}M trainable parameters.')
    logger.info(f'The mask_decoder model has {count_parameters(mask_decoder):,}M trainable parameters.')

    encoder_opt = AdamW([i for i in img_encoder.parameters() if i.requires_grad==True], lr=args.lr, weight_decay=0)
    encoder_scheduler = torch.optim.lr_scheduler.LinearLR(encoder_opt, start_factor=1.0, end_factor=0.01, total_iters=args.max_epoch)
    feature_opt = AdamW(prompt_encoder.parameters(), lr=args.lr, weight_decay=0)
    feature_scheduler = torch.optim.lr_scheduler.LinearLR(feature_opt, start_factor=1.0, end_factor=0.01,
                                                          total_iters=args.max_epoch)
    decoder_opt = AdamW([i for i in mask_decoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0)
    decoder_scheduler = torch.optim.lr_scheduler.LinearLR(decoder_opt, start_factor=1.0, end_factor=0.01, total_iters=args.max_epoch)
    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    loss_cal = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
    best_loss = np.inf
    patch_size = args.rand_crop_size[0]
    
    debug_time = True
    
    for epoch_num in range(args.max_epoch):
        loss_summary = []
        img_encoder.train()
        prompt_encoder.train()
        mask_decoder.train()
        
        if debug_time:
            batch_end = time.time()

        for idx, (img, seg, spacing) in enumerate(train_data):
            if debug_time:
                batch_start = time.time()
                print("data loading spend time", batch_start - batch_end)
            # print('seg: ', seg.sum())
            out = F.interpolate(img.float(), scale_factor=input_image_size / patch_size, mode='trilinear')  # TODO: 256 input
            # input_batch = (out.cuda() - pixel_mean) / pixel_std
            input_batch = out.to(device)
            batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
            input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)
            
            batch_features = img_encoder(input_batch, batchsize)  # [2, 16, 16, 16, 768]

            use_click = random.randint(0, 1)
            if use_click:
                # randomly sample click points
                points_torch = []
                labels_torch = []
                
                click_num = random.randint(1, args.max_clicks)
                
                for _seg in seg:
                    _seg = _seg.unsqueeze(0)
                    l = len(torch.where(_seg == 1)[0])
                    this_points = []
                    this_labels = []
                    if l > 0:
                        sample = np.random.choice(np.arange(l), click_num, replace=True)
                        x = torch.where(_seg == 1)[1][sample].unsqueeze(1)
                        y = torch.where(_seg == 1)[2][sample].unsqueeze(1)
                        z = torch.where(_seg == 1)[3][sample].unsqueeze(1)
                        points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
                        this_points.append(points.to(device).transpose(0,1))  # 1, num_click, 3
                        this_labels.append(torch.ones(1, click_num))  # 1, num_click
                        
                        l = len(torch.where(_seg == 0)[0])
                        sample = np.random.choice(np.arange(l), click_num, replace=True)
                        x = torch.where(_seg == 0)[1][sample].unsqueeze(1)
                        y = torch.where(_seg == 0)[2][sample].unsqueeze(1)
                        z = torch.where(_seg == 0)[3][sample].unsqueeze(1)
                        points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
                        this_points.append(points.to(device).transpose(0,1))
                        this_labels.append(torch.zeros(1, click_num))
                        
                    else:                    
                        l = len(torch.where(_seg == 0)[0])
                        sample = np.random.choice(np.arange(l), click_num * 2, replace=True)
                        x = torch.where(_seg == 0)[1][sample].unsqueeze(1)
                        y = torch.where(_seg == 0)[2][sample].unsqueeze(1)
                        z = torch.where(_seg == 0)[3][sample].unsqueeze(1)
                        points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
                        this_points.append(points.to(device).transpose(0,1))
                        this_labels.append(torch.zeros(1, click_num * 2))
                    
                    this_points = torch.cat(this_points, 1)  #  1, num_click * 2, 3
                    this_labels = torch.cat(this_labels, 1)  #  1, num_click * 2
                    
                    points_torch.append(this_points)
                    labels_torch.append(this_labels)
                
                points_torch = torch.cat(points_torch, 0)  #  bs, num_click * 2, 3
                labels_torch = torch.cat(labels_torch, 0)  #  bs, num_click * 2
                prompt_input = [points_torch, labels_torch]
                
                point_feature, _dense = prompt_encoder(prompt_input, None, None)
                
            else:
                point_feature = None
            image_pe = prompt_encoder.get_dense_pe().to(batch_features.device)
            
            masks = mask_decoder(batch_features, image_pe, point_feature, None, False)

            # TODO: check if this permute is needed
            masks = masks.permute(0, 1, 4, 2, 3)
            seg = seg.to(device)
            seg = seg.unsqueeze(1)
            loss = loss_cal(masks, seg)
            loss_summary.append(loss.detach().cpu().numpy())
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            feature_opt.zero_grad()
            loss.backward()
            logger.info(
                'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(train_data)) + ": loss:" + str(
                    loss_summary[-1].flatten()[0]))
            torch.nn.utils.clip_grad_norm_(img_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(mask_decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(prompt_encoder.parameters(), 1.0)
            encoder_opt.step()
            feature_opt.step()
            decoder_opt.step()
            
            if debug_time:
                batch_end = time.time()
                print("batch spend time", batch_end - batch_start)
                
        encoder_scheduler.step()
        feature_scheduler.step()
        decoder_scheduler.step()

        logger.info("- Train metrics: " + str(np.mean(loss_summary)))

        img_encoder.eval()
        prompt_encoder.eval()
        mask_decoder.eval()
        with torch.no_grad():
            loss_summary = []
            for idx, (img, seg, spacing, path) in enumerate(val_data):
                # print('seg: ', seg.sum())
                out = F.interpolate(img.float(), scale_factor=input_image_size / patch_size, mode='trilinear')  # TODO: for 256
                input_batch = out.to(device)
                batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
                input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)
                
                batch_features = img_encoder(input_batch, batchsize)  # [2, 16, 16, 16, 768]
                
                # randomly sample click points
                # randomly sample click points
                points_torch = []
                labels_torch = []
                
                click_num = 5
                
                for _seg in seg:
                    _seg = _seg.unsqueeze(0)
                    l = len(torch.where(_seg == 1)[0])
                    this_points = []
                    this_labels = []
                    if l > 0:
                        sample = np.random.choice(np.arange(l), click_num, replace=True)
                        x = torch.where(_seg == 1)[1][sample].unsqueeze(1)
                        y = torch.where(_seg == 1)[2][sample].unsqueeze(1)
                        z = torch.where(_seg == 1)[3][sample].unsqueeze(1)
                        points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
                        this_points.append(points.to(device).transpose(0,1))
                        this_labels.append(torch.ones(1, click_num))
                        
                        l = len(torch.where(_seg == 0)[0])
                        sample = np.random.choice(np.arange(l), click_num, replace=True)
                        x = torch.where(_seg == 0)[1][sample].unsqueeze(1)
                        y = torch.where(_seg == 0)[2][sample].unsqueeze(1)
                        z = torch.where(_seg == 0)[3][sample].unsqueeze(1)
                        points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
                        this_points.append(points.to(device).transpose(0,1))
                        this_labels.append(torch.zeros(1, click_num))
                        
                    else:
                        l = len(torch.where(_seg == 0)[0])
                        sample = np.random.choice(np.arange(l), click_num * 2, replace=True)
                        x = torch.where(_seg == 0)[1][sample].unsqueeze(1)
                        y = torch.where(_seg == 0)[2][sample].unsqueeze(1)
                        z = torch.where(_seg == 0)[3][sample].unsqueeze(1)
                        points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
                        this_points.append(points.to(device).transpose(0,1))
                        this_labels.append(torch.zeros(1, click_num * 2))
                    
                    this_points = torch.cat(this_points, 1)
                    this_labels = torch.cat(this_labels, 1)
                    
                    points_torch.append(this_points)
                    labels_torch.append(this_labels)
                
                points_torch = torch.cat(points_torch, 0)
                labels_torch = torch.cat(labels_torch, 0)
                prompt_input = [points_torch, labels_torch]

                image_pe = prompt_encoder.get_dense_pe().to(batch_features.device)
                point_feature, _dense = prompt_encoder(prompt_input, None, None)
                masks = mask_decoder(batch_features, image_pe, point_feature, None, True)
                
                masks = masks.permute(0, 1, 4, 2, 3)
                seg = seg.to(device)
                seg = seg.unsqueeze(1)
                loss = dice_loss(masks, seg)
                loss_summary.append(loss.detach().cpu().numpy())
                logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(val_data)) + ": loss:" + str(
                        loss_summary[-1].flatten()[0]))
        logger.info("- Val metrics: " + str(np.mean(loss_summary)))


        is_best = False
        if np.mean(loss_summary) < best_loss:
            best_loss = np.mean(loss_summary)
            is_best = True
            
        img_encoder.train()  # very tricky
        # NOTE: The devil lies in the detail:
        # Calling model.eval() will trigger the merging of LoRA parameters with the corresponding pretrained ones, which eliminates additional latency for subsequent forward passes. Calling model.train() again will undo the merge. This can be disabled by passing merge_weights=False to LoRA layers.
        save_checkpoint({"epoch": epoch_num + 1,
                        "best_val_loss": best_loss,
                         "encoder_dict": img_encoder.state_dict(),
                         "decoder_dict": mask_decoder.state_dict(),
                         "feature_dict": prompt_encoder.state_dict(),
                         "encoder_opt": encoder_opt.state_dict(),
                         "feature_opt": feature_opt.state_dict(),
                         "decoder_opt": decoder_opt.state_dict()
                         },
                        is_best=is_best,
                        checkpoint=args.snapshot_path)
        logger.info("- Val metrics best: " + str(best_loss))


if __name__ == "__main__":
    main()

