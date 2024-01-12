import torch
from torch.utils.data import Dataset 
import scipy.io as scio
import numpy as np
from torch import nn 
import logging 
import time 
import os
import os.path as osp
import cv2
import math 
import albumentations

class TrainData(Dataset):
    def __init__(self,args):
        self.data_dir= args.train_data_path
        self.data_list = os.listdir(args.train_data_path)
        self.img_files = []
        
        self.ratio = args.B
        self.resize_w, self.resize_h = args.size
        for image_dir in os.listdir(args.train_data_path):
            train_data_path = os.path.join(args.train_data_path,image_dir)
            data_path = os.listdir(train_data_path)
            data_path.sort()
            for sub_index in range(len(data_path)-self.ratio):
                sub_data_path = data_path[sub_index:]
                meas_list = []
                count = 0
                for image_name in sub_data_path:
                    meas_list.append(os.path.join(train_data_path,image_name))
                    if (count+1)%self.ratio==0:
                        self.img_files.append(meas_list)
                        meas_list = []
                    count += 1
        
    def __getitem__(self,index):
        gt = np.zeros([self.ratio, self.resize_h, self.resize_w],dtype=np.float32)
        # meas = np.zeros([self.resize_h, self.resize_w],dtype=np.float32)
        gt_images_list = []
        p = np.random.randint(0,10)>5
        image = cv2.imread(self.img_files[index][0])
        image_h, image_w = image.shape[:2]
        # mask_h,mask_w= self.mask.shape[1:]
        
        crop_h = np.random.randint(self.resize_h//2,image_h)
        crop_w = np.random.randint(self.resize_w//2,image_w)
        crop_p = np.random.randint(0,10)>5
        flip_p = np.random.randint(0,10)>5
        transform = albumentations.Compose([
            albumentations.CenterCrop(height=crop_h,width=crop_w,p=crop_p),
            albumentations.HorizontalFlip(p=flip_p),
            albumentations.Resize(self.resize_h,self.resize_w)
        ])
        rotate_flag = np.random.randint(0,10)>5
        for i,image_path in enumerate(self.img_files[index]):
            image = cv2.imread(image_path)

            transformed = transform(image=image)
            image = transformed['image']
            # print(rotate_flag)
            if rotate_flag:
                image = cv2.flip(image, 1)
                image = cv2.transpose(image)
            pic_t = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)[:,:,0]
            pic_t = pic_t.astype(np.float32)
            pic_t /= 255.
            gt_images_list.append(pic_t)
            # mask_t = self.mask[i, :, :]
            gt[i, :, :] = pic_t
            # meas += np.multiply(mask_t.numpy(), pic_t)
        return gt
    def __len__(self,):
        return len(self.img_files)

class TestData(Dataset):
    def __init__(self,args):
        self.data_path = args.benchmark_path
        self.data_list = os.listdir(self.data_path)
        self.cr = args.B 

    def __getitem__(self,index):
        pic = scio.loadmat(os.path.join(self.data_path,self.data_list[index]))
        if 'orig' in pic:
            pic = pic['orig']
        elif 'patch_save' in pic:
            pic = pic['patch_save']
        elif 'p1' in pic:
            pic = pic['p1']
        elif 'p2' in pic:
            pic = pic['p2']
        elif 'p3' in pic:
            pic = pic['p3']
        pic = pic / 255

        pic_gt = np.zeros([pic.shape[2] // self.cr, self.cr, 256, 256])
        for jj in range(pic.shape[2]):
            if jj // self.cr>=pic_gt.shape[0]:
                break
            if jj % self.cr == 0:
                n = 0
            pic_t = pic[:, :, jj]

            pic_gt[jj // self.cr, n, :, :] = pic_t
            n += 1
        return pic_gt
    def __len__(self,):
        return len(self.data_list)

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def compare_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))



def compare_psnr(img1, img2, shave_border=0):
    height, width = img1.shape[:2]
    img1 = img1[shave_border:height - shave_border, shave_border:width - shave_border]
    img2 = img2[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = img1 - img2
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def save_image(out,gt,image_name,show_flag=False):
    sing_out = out.transpose(1,0,2).reshape(out.shape[1],-1)
    if gt is None:
        result_img = sing_out*255
    else:
        sing_gt = gt.transpose(1,0,2).reshape(gt.shape[1],-1)
        result_img = np.concatenate([sing_out,sing_gt],axis=0)*255
    result_img = result_img.astype(np.float32)
    cv2.imwrite(image_name,result_img)
    if show_flag:
        cv2.namedWindow('image',0)
        cv2.imshow('image',result_img.astype(np.uint8))
        cv2.waitKey(0)

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

def save_real_image(out,image_dir):
    out = out*255
    if not osp.exists(image_dir):
        os.makedirs(image_dir)
    for i in range(out.shape[0]):
        image = out[i]
        cv2.imwrite(osp.join(image_dir,str(i)+".png"),image)

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

def Logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(filename)s [line: %(lineno)s] - %(message)s')

    localtime = time.strftime('%Y_%m_%d_%H_%M_%S')
    logfile = os.path.join(log_dir,localtime+'.log')
    fh = logging.FileHandler(logfile,mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger 


# x = [H, W], ratios = [B, H, W] 
def integer_split_with_ratio(x, ratio):
    assert (x >= 0).all()
    assert (ratio >= 0).all()
    assert (ratio.sum(0) > 0).all()
    remainder = x
    ratio, idx = torch.sort(ratio,dim=0)
    ratio_cumsum = torch.cumsum(ratio.flip([0]),dim=0).flip([0])
    fracs = torch.where(ratio_cumsum != 0, torch.div(ratio,ratio_cumsum), torch.ones_like(ratio))
    parts = torch.zeros_like(fracs)
    # portions = torch.zeros_like(fracs)
    for i in range(ratio.shape[0]):
        parts[i] = torch.round(remainder * fracs[i])
        remainder = remainder - parts[i]
        # if i == ratio.shape[0] -2 and (x==remainder).any():
        #     parts[i] = torch.where(x==remainder, torch.ones_like(x).to(x.device), parts[i])
        #     remainder = torch.where(x==remainder,  remainder - torch.ones_like(x).to(x.device), remainder)
    # portions = parts.gather(0, idx.argsort(0)
    # assert (parts.sum(0).eq(x)).all()

    return parts.gather(0, idx.argsort(0))


def discretize1(x, B=8, gray_levels=16, mask_type='discrete_gray'):
    if mask_type == 'binary_dense': 
        y = 0.5*(x.sign() + 1)
        y[y == 0.5] = 1
    if mask_type == 'binary_sparse':
        y = torch.floor(x*B)
        y[y == B] = B-1
    if mask_type == 'discrete_gray':

        y = torch.round(x*gray_levels) / gray_levels
        y[y == 1] = (gray_levels-1) / gray_levels
        y[:,y.sum(0)==0] = 1 / gray_levels

        for i in range(y.shape[1]):
            for j in range(y.shape[2]):       
                if torch.count_nonzero(y[:,i,j])<2:
                    a,idx = torch.sort(y[:,i,j])
                    y[:,i,j] = torch.zeros_like(y[:,i,j])
                    y[idx[-1],i,j] = (gray_levels-1) / gray_levels
                    y[idx[-2],i,j] = 1 / gray_levels

        y1 = y

        if ~torch.isin(y, torch.range(0,gray_levels-1).to(y.device)/gray_levels).all():
            print('there is a mask value out of range after discretization!')
            quit()

        residual = (y.sum(0) - 1)*gray_levels
        residual_sign = torch.sign(residual)
        delta = integer_split_with_ratio(torch.abs(residual), torch.div(y,y.sum(0)))
        y = y - residual_sign*delta / gray_levels

        if ~torch.isin(y, torch.range(0,gray_levels-1).to(y.device)/gray_levels).all():
            print('there is a mask value out of range after structuralization!')
            print('the number of 1 : {}'.format(torch.numel(y) - torch.count_nonzero(y-1)))
            for i in range(y.shape[1]):
                for j in range(y.shape[2]):
                    if torch.isin(y[:,i,j], torch.tensor([0,1]).to(y.device)).all():
                        print("orgi:{}, {}, {}".format(i,j,x[:,i,j]))
                        print("disc:{}, {}, {}".format(i,j,y1[:,i,j]))
                        print("stru:{}, {}, {}".format(i,j,y[:,i,j]))
                        print("\n")

    return y


def discretize(x, B=8, gray_levels=16, mask_type='discrete_gray'):
    if mask_type == 'binary_dense':
        y = 0.5*(x.sign() + 1)
        y[y == 0.5] = 1
    if mask_type == 'binary_sparse':
        y = torch.floor(x*B)
        y[y == B] = B-1
    if mask_type == 'discrete_gray':
        tiny = torch.finfo(x.dtype).tiny

        y = torch.round(x*gray_levels) / gray_levels
        y[y == 1] = (gray_levels-1) / gray_levels
        y[:,y.sum(dim=0)==0] = 1 / gray_levels
        # y[:,y.sum(dim=0)==0] = (torch.round(torch.tensor(1 / B) * gray_levels) / gray_levels).to(y.device)
        # y[:,y.sum(dim=0)==0] = (torch.ceil(torch.tensor(1 / B) * gray_levels) / gray_levels).to(y.device)

        # if torch.min(y) < 0 and torch.max(y) > (gray_levels-1) / gray_levels:
        #     print('there is a error!')
        #     quit()
 
        y_sum = y.sum(dim=0)
        dev = gray_levels*(y_sum - 1)
        weights = gray_levels*y

        total_weights = weights.sum(dim=0)
        for i in range(B):
            p = weights[i] / torch.maximum(total_weights, tiny*torch.ones(total_weights.size()).to(total_weights))
            delta = torch.round(p*dev)
            y[i] = y[i] - (delta / gray_levels)
            total_weights -= weights[i]
            dev -= delta  
   
    return y

class SignZero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input == 0
        return output.float()

    def backward(ctx, grad_output):
        grad_input = torch.clamp(grad_output, -1, 1)
        return grad_input

findzero = SignZero.apply


def checkpoint(epoch, model, optimizer, model_out_path):
    for p in list(model.parameters()):
        if hasattr(p, 'org'):
           continuous_mask = p.org.detach().clone()
    torch.save({'pretrain_epoch':epoch,
                'continuous_mask':continuous_mask,
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict()}, model_out_path)


def load_checkpoint(model, pretrained_dict, optimizer=None):
    model_dict = model.state_dict()
    pretrained_model_dict = pretrained_dict['state_dict']
    load_dict = {k: p for k, p in pretrained_model_dict.items() if k in model_dict.keys()} # filtering parameters
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    for p in list(model.parameters()):
        if hasattr(p, 'org'):
            if 'continuous_mask' in pretrained_dict:
                p.org = pretrained_dict['continuous_mask']
                print('Find available continuous mask!')
            else:
                p.org = p.detach().clone()
                print('No available continuous mask, use the latest discrete mask as continuous mask!')
    if optimizer is not None:
        optimizer.load_state_dict(pretrained_dict['optimizer']) #loading pretrained optimizer when network is not changed.
    print('Model parameter number: {}, Pretrained parameter number: {}, Loaded parameter number: {}'\
        .format(len(model_dict), len(pretrained_model_dict), len(load_dict)))

def load_checkpoint1(model,pretrained_dict,strict=False):
    if strict is True:
        try: 
            model.load_state_dict(pretrained_dict)
        except:
            print("load model error!")
    else:
        model_dict = model.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        for k in pretrained_dict: 
            if model_dict[k].shape != pretrained_dict[k].shape:
                pretrained_dict[k] = model_dict[k]
                print("layer: {} parameters size is not same!".format(k))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict,strict=False)