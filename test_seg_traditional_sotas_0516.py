import os
from dataset.datasets import load_data_volume
import argparse

import torch
import numpy as np
import logging
from utils.model_util import get_model
import torch.nn.functional as F
from monai.metrics import HausdorffDistanceMetric
import time

from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference

from utils.util import setup_logger
import surface_distance
from surface_distance import metrics

import ast
import matplotlib.pyplot as plt

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

    plt.savefig(f"exp_images_comp/exp_images_unetr++/{name}.png", dpi=120)
    plt.clf()

def set_default_arguments(args):
    args.rand_crop_size = (128, 128, 128)
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str
    )
    parser.add_argument(
        "--snapshot_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--method",
        default=None,
        type=str,
        choices=["swin_unetr", "swin_unetr2", "unetr", "3d_uxnet", "nnformer", "unetr++", "transbts", "DAF3D"],
    )
    parser.add_argument(
        "--data_prefix",
        default="",
        type=str,
    )
    parser.add_argument("--overlap", default=0.7, type=float)
    parser.add_argument(
        "--infer_mode", default="constant", type=str, choices=["constant", "gaussian"]
    )
    parser.add_argument("--save_pred", action="store_true")
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument("--tolerance", default=5, type=int)

    args = parser.parse_args()
    args = set_default_arguments(args)
    
    #args.weight_path = os.path.join(args.snapshot_path, "['lung','lung2','Lung421','pancreas','kits23','hepatic']")
    args.weight_path = os.path.join(args.snapshot_path, args.data)
    
    
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    setup_logger(logger_name="test", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"test")
    logger.info(str(args))
    logger.info("load weight:"+str(args.weight_path))

    args.data = ast.literal_eval(args.data)
    args.data_prefix = [f"../datafile/{dataset_name}" for dataset_name in args.data]
    print(args.data_prefix)
    
    test_data = load_data_volume(
        datas=args.data,
        path_prefixes=args.data_prefix,
        batch_size=1,
        augmentation=False,
        split="test",  
        deterministic=True,
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker = args.num_worker
    )

    seg_net = get_model(args).cuda()
    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean", get_not_nans=True)
    
    
    ckpt = torch.load(os.path.join(args.weight_path, "best_debug.pth.tar"))

    seg_net.load_state_dict(ckpt["encoder_dict"])
    logger.info(f"Loading checkpoint done!")
    logger.info("Best validation Dice: {:.6f}".format(1 - ckpt["best_val_loss"]))

    logger.info(
        "#Param: {}".format(sum(p.numel() for p in seg_net.parameters() if p.requires_grad))
    )

    loss_summary = []
    nsd_list = []

    
    save_pred_path = os.path.join(args.snapshot_path, "predictions")
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)

    hd95_overview = []
    loss_overview = []
    mean_overview = []
    
    total_infer_time = 0
    infer_num = 0
    
    # from thop import profile
    # input_batch = torch.randn((1, 1, 128, 128, 128)).to('cuda')
    # macs_img_encoder, _ = profile(seg_net, inputs=(input_batch, ))
    # print("img_encoder MAC数量(G):", macs_img_encoder/1e9)
    # from calculate_flops.calflops.flops_counter import calculate_flops

    # flops, macs, params = calculate_flops(model=seg_net, 
    #                                   input_shape=(1, 1, 128, 128, 128),
    #                                   output_as_string=True,
    #                                   output_precision=4)
    # print("method", args.method)
    # print("macs", macs)
    # print("params", params)
    # raise "s"
    
    with torch.no_grad():
        seg_net.eval()
        for idx, (img, seg, spacing, path) in enumerate(test_data):
            
            img = img.cuda().float()
            img = img[:, :1, :, :, :]
            start_time = time.time()
            masks = seg_net(img)

            
            
            end_time = time.time()
            total_infer_time += end_time - start_time
            infer_num += len(masks)
            
            seg = seg.unsqueeze(1).cuda()
            loss = dice_loss(masks, seg)
            loss_summary.append(loss.detach().cpu().numpy())
            #print(path)
            masks = F.softmax(masks, dim=1)#[:, 1]
            masks = masks > 0.5
            hd95 = hd95_metric(masks, seg)

            masks = masks[:, 1]
            torch.save(
                img[0].cpu(),
                os.path.join(save_pred_path, f"img_{idx}.pt")
            )
            torch.save(
                masks[0].cpu(),
                os.path.join(save_pred_path, f"mask_{idx}.pt")
            )
            loss_overview.append(loss.mean().detach().cpu().numpy())
            mean_overview.append(img.mean().detach().cpu().numpy().squeeze())
            if not torch.isnan(hd95):
                hd95_overview.append(hd95.mean().detach().cpu().numpy())
            
            
            logger.info(
                " Case {} - Dice {:.6f} ".format(
                    path[0].split("/")[-1], loss.item()
                )
            )
    logger.info(
        "- Dice: {:.6f} - {:.6f} | NSD: {:.6f} - {:.6f}".format(
            np.mean(loss_summary), np.std(loss_summary), np.mean(nsd_list), np.std(nsd_list)
        )
    )
    
    print(args.method)
    #print("img_encoder MAC数量(G):", macs_img_encoder/1e9)
    print("平均推理时间", round(total_infer_time / infer_num, 4)) 
    logger.info("- Val metrics: " + str(np.mean(loss_summary)))
    
    print("dice")
    print((1 - sum(loss_overview) / len(loss_overview)))
    print("hd95")
    print(sum(hd95_overview) / len(hd95_overview))


if __name__ == "__main__":
    main()
