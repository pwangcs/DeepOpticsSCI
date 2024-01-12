import argparse 
import torch 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--pretrained_model_path', default='/home/wangping/codes/iccv2023/weights/SCI-Res2former/2023_08_15_02_13_11/epoch_56.pth', type=str)
    parser.add_argument("--test_weight_path", default='/home/wangping/codes/iccv2023/weights/E2E-STFormer-16graylevels/2023_02_20_14_30_10/epoch_100.pth', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--encoder_lr', default=0.0001, type=float)
    parser.add_argument('--decoder_lr', default=0.0001, type=float)
    parser.add_argument('--gray_levels', default=16, type=int)
    parser.add_argument('--B', default=8, type=int)
    parser.add_argument('--size', default=[256,256])
    parser.add_argument('--train_type', default='sim', type=str, help='train for simulated data [sim] or real data [real]')
    parser.add_argument('--decoder_type', default='STFormer', type=str, help='reconstruction network type: Unet, RevSCI, ConvFormer, UnetSCI, STFormer')
    parser.add_argument('--color_channels', default=1,type=int)
    parser.add_argument('--save_model_step', default=1, type=int)
    parser.add_argument('--save_train_image_step', default=500, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--iter_step', default=400, type=int)
    parser.add_argument('--test_flag', default=False, type=bool)
    parser.add_argument('--train_data_path',type=str,default='/home/wangping/datasets/DAVIS/JPEGImages/480p')
    parser.add_argument('--benchmark_path',type=str,default='/home/wangping/datasets/SCI/simulation')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--torchcompile',nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")

    args = parser.parse_args()
    return args

