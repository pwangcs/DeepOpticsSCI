from torch.utils.data import DataLoader 
import torch 
import os
import os.path as osp
import scipy.io as scio
import numpy as np 
import einops
from utils import save_image, compare_ssim, compare_psnr, TestData
from opts import parse_args
from models.network import E2E_SCI
from utils import Logger, load_checkpoint, load_checkpoint_128_block_256, load_checkpoint_128_expand_256,save_single_image
import time
import cv2



def save_single_image(images,image_dir,batch,name="",demosaic=False):
    images = images*255
    if len(images.shape)==4:
        frames = images.shape[1]
    else:
        frames = images.shape[0]
    for i in range(frames):
        begin_frame = batch*frames
        if len(images.shape)==4:
            single_image = images[:,i].transpose(1,2,0)[:,:,::-1]
        else:
            single_image = images[i]
        if demosaic:
            single_image = demosaicing_bayer(single_image,pattern='BGGR')
        cv2.imwrite(osp.join(image_dir,name+"_"+str(begin_frame+i+1)+".png"),single_image)
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
            # gt_temp = einops.rearrange(gt,"b f h w-> (b f) h w")
            out_pic_list = network(torch.from_numpy(gt).to(args.device))
        # out_pic_list = network(meas)
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
        
    # all_out = np.array(all_out)
    # all_gt= np.array(all_gt)
    # print(all_out.shape)
    # print(all_gt.shape)
    # out_gt = np.stack([all_out,all_gt],axis=0)
    # scio.savemat("/home/wangping/codes/iccv2023/all_mat/res.mat",{"res":out_gt})

    for i,name in enumerate(test_data.data_list):
        _name,_ = name.split("_")
        psnr_dict[_name] = psnr_list[i]
        ssim_dict[_name] = ssim_list[i]
        out = out_list[i]
        gt = gt_list[i]
        for j in range(out.shape[0]):
            image_name = os.path.join(test_dir,"epoch_"+str(epoch)+"_"+_name+"_"+str(j)+".png")
            save_image(out[j],gt[j],image_name)
            # image_dir = os.path.join("/home/wangping/codes/iccv2023/stformer_16_epoch_280",_name)
            # if not os.path.exists(image_dir):
            #     os.makedirs(image_dir)
            # save_single_image(out[j],image_dir,j,_name)
    if writer is not None:
        writer.add_scalar("psnr_mean",np.mean(psnr_list),epoch)
        writer.add_scalar("ssim_mean",np.mean(ssim_list),epoch)
    if logger is not None:
        logger.info("psnr_mean: {:.4f}.".format(np.mean(psnr_list)))
        logger.info("ssim_mean: {:.4f}.".format(np.mean(ssim_list)))
    return psnr_dict, ssim_dict

if __name__=="__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    # args.test_weight_path = '/home/wangping/codes/iccv2023/weights/E2E-STFormer-16graylevels/2023_02_20_14_30_10/epoch_100.pth'
    # args.test_weight_path = '/home/wangping/codes/iccv2023/weights/E2E-UnetSCI-16graylevels/2023_02_21_21_34_08/epoch_200.pth'

    # args.test_weight_path = '/home/wangping/codes/iccv2023/checkpoints/stformer_256_cr_10_epoch_138.pth'
    # args.decoder_type = 'STFormer'
    # args.test_weight_path = '/home/wangping/codes/iccv2023/checkpoints/final_simulation/stformer_16_epoch_280.pth'
    args.decoder_type = 'Res2former'
    args.test_weight_path = '/home/wangping/codes/iccv2023/checkpoints/final_simulation/unetsci_16_epoch_342.pth'
    # args.decoder_type = 'UnetSCI_L'
    # args.test_weight_path = '/home/wangping/codes/iccv2023/checkpoints/final_simulation/unetsci_large_16_epoch_261.pth'
    # args.benchmark_path = '/home/wangping/datasets/SCI/simulation'
    # args.benchmark_path = '/home/wangping/datasets/SCI/simulation1'

    args.size = [256,256]
    args.train_type = 'sim'
    args.B = 8

    test_path = "test_results"
    network = E2E_SCI(args).to(args.device)
    log_dir = os.path.join("test_results","log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir,exist_ok=True)
    logger = Logger(log_dir)
    if args.test_weight_path is not None:
        pretrained_dict = torch.load(args.test_weight_path)
        load_checkpoint(network, pretrained_dict)
        # load_checkpoint_128_block_256(network, pretrained_dict)
        if 'pretrain_epoch' in pretrained_dict.keys():
            pretrain_epoch = pretrained_dict['pretrain_epoch']
            logger.info('Pretrain epoch: {}'.format(pretrain_epoch))     
    else:
        raise ValueError('Please input a weight path for testing.')

    psnr_dict, ssim_dict = test(args, network, logger, test_path)
    logger.info("psnr: {}.".format(psnr_dict))
    logger.info("ssim: {}.".format(ssim_dict))
