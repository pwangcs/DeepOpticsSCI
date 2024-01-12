import torch 
import os 
import os.path as osp
import cv2 
import scipy.io as scio 
import numpy as np 
import argparse
from model.digital_layer import STFormer, Res2former
from utils import Logger, save_real_image

os.environ["CUDA_VISIBLE_DEVICES"] = "8"

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

parser = argparse.ArgumentParser(description='configuration')
parser.add_argument('--configure', default='stformer_random_mask', type=str,
                    help='stformer_random_mask, stformer_structural_mask, res2former_structural_mask')
parser.add_argument('--device', default='cuda', type=str)
args = parser.parse_args()


def test(network, mask):
    network = network.eval()  
    out_list= []
    meas_dir = "./testdata/real_data" + "/" + args.configure + "/" + "meas"
    result_dir = "./testdata/real_data" + "/" + args.configure + "/" + "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for meas_name in os.listdir(meas_dir):
        meas_path = osp.join(meas_dir,meas_name)
        meas = cv2.imread(meas_path,0)

        meas = meas[3:-3,4:-4]  # detele invalid pixels at the boundary, caused by the hardware defects of DMD

        with torch.no_grad():
            meas = meas.astype(np.float32)
            meas = 5*meas/255. if args.configure == 'stformer_random_mask' else meas/255.            
            meas = torch.from_numpy(meas).unsqueeze(0)
            meas = meas.to(args.device)
            out_pic_list = network(meas, mask)
        out_pic = out_pic_list[-1].cpu().numpy()
        print(out_pic.shape)
        out_list.append(out_pic)

    for i,name in enumerate(os.listdir(meas_dir)):
        _name,_ = name.split(".")
        out = out_list[i]
        for j in range(out.shape[0]):
            image_dir = os.path.join(result_dir,_name)
            save_real_image(out[j],image_dir)
    

if __name__=="__main__":
    if args.configure == 'stformer_random_mask':
        network = STFormer(color_channels=1,units=4,dim=64,frames=10).to(args.device)
        mask = scio.loadmat('testdata/real_data/stformer_random_mask/random_mask.mat')['mask']
        weight_path = 'weight/real_stformer_random_mask.pth'
    if args.configure == 'stformer_structural_mask':
        network = STFormer(color_channels=1,units=4,dim=64,frames=10).to(args.device)
        mask = scio.loadmat('testdata/real_data/stformer_structural_mask/learned_mask.mat')['mask']
        weight_path = 'weight/real_stformer_structural_mask.pth'
    elif args.configure == 'res2former_structural_mask':
        network = Res2former(dim=96,stage_num=1,depth_num=[3,3],color_ch=1).to(args.device)
        mask = scio.loadmat('testdata/real_data/res2former_structural_mask/learned_mask.mat')['mask']
        weight_path = 'weight/real_res2former_structural_mask.pth'
    else:
        TypeError('Reconstruction method has never been undefined.')

    mask = mask[:,3:-3,4:-4]   # detele invalid pixels at the boundary, caused by the hardware defects of DMD
    mask = torch.from_numpy(mask).float().to(args.device)

    # load model weights
    model_dict = network.state_dict()
    pretrained_dict = torch.load(weight_path)
    model_state_dict =  pretrained_dict["state_dict"]
    load_dict = {k:v for k,v in model_state_dict.items() if k in model_dict}
    model_dict.update(load_dict)
    network.load_state_dict(model_dict)
    
    # inference
    test(network, mask)