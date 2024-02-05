from torch.utils.data import DataLoader 
import torch 
import os
import os.path as osp
import scipy.io as scio
import numpy as np 
import einops
from utils import Logger, load_checkpoint, save_image, compare_ssim, compare_psnr, TestData
from opts import parse_args
from model.joint_optimization import DeepOpticsSCI
import time
import cv2


def test(args, network, logger, test_dir, writer=None, epoch=1):
    network = network.eval()
    test_data = TestData(args) 
    test_data_loader = DataLoader(test_data, shuffle=False, batch_size=1)    

    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    out_list,gt_list = [],[]
    all_out = []
    all_gt = []
    runtime = 0
    for iter, data in enumerate(test_data_loader):
        gt = data 
        gt = gt[0].float().numpy()
        batch_size,frames,height,width = gt.shape
        with torch.no_grad():
            out_pic_list = network(torch.from_numpy(gt).to(args.device))
        out_pic = out_pic_list[-1].cpu().numpy()
        psnr_t = 0
        ssim_t = 0
        for ii in range(batch_size):
            for jj in range(frames):
                out_pic_p = out_pic[ii,jj, :, :]
                gt_t = gt[ii,jj, :, :]
                all_out.append(out_pic_p)
                all_gt.append(gt_t) 
                psnr_t += compare_psnr(gt_t*255,out_pic_p*255)
                ssim_t += compare_ssim(gt_t*255,out_pic_p*255)
        psnr = psnr_t / (batch_size* frames)
        ssim = ssim_t / (batch_size* frames)
        psnr_list.append(np.round(psnr,4))
        ssim_list.append(np.round(ssim,4))
        out_list.append(out_pic)
        gt_list.append(gt)
        

    for i,name in enumerate(test_data.data_list):
        _name,_ = name.split("_")
        psnr_dict[_name] = psnr_list[i]
        ssim_dict[_name] = ssim_list[i]
        out = out_list[i]
        gt = gt_list[i]
        for j in range(out.shape[0]):
            image_name = os.path.join(test_dir,"epoch_"+str(epoch)+"_"+_name+"_"+str(j)+".png")
            save_image(out[j],gt[j],image_name)
    if writer is not None:
        writer.add_scalar("psnr_mean",np.mean(psnr_list),epoch)
        writer.add_scalar("ssim_mean",np.mean(ssim_list),epoch)
    if logger is not None:
        logger.info("psnr_mean: {:.4f}.".format(np.mean(psnr_list)))
        logger.info("ssim_mean: {:.4f}.".format(np.mean(ssim_list)))
    return psnr_dict, ssim_dict

if __name__=="__main__":
    args = parse_args()
    test_path = "test_results"
    network = DeepOpticsSCI(args).to(args.device)
    log_dir = os.path.join("test_results","log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir,exist_ok=True)
    logger = Logger(log_dir)
    if args.test_weight_path is not None:
        pretrained_dict = torch.load(args.test_weight_path)
        load_checkpoint(network, pretrained_dict) 
    else:
        raise ValueError('Please input a weight path for testing.')

    psnr_dict, ssim_dict = test(args, network, logger, test_path)
    logger.info("psnr: {}.".format(psnr_dict))
    logger.info("ssim: {}.".format(ssim_dict))
