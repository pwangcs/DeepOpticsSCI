import argparse 
import torch 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--pretrained_model_path', default=None, type=str, help='pre-trained weight path for fine-tuning')
    parser.add_argument("--test_weight_path", default='weight/res2former_4bit_structural_mask.pth', type=str,  help='pre-trained weight path for testing')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--encoder_lr', default=0.0001, type=float)
    parser.add_argument('--decoder_lr', default=0.0001, type=float)
    parser.add_argument('--gray_levels', default=16, type=int)
    parser.add_argument('--B', default=8, type=int)
    parser.add_argument('--size', default=[256,256])
    parser.add_argument('--train_type', default='sim', type=str, help='train for simulated data [sim] or real data [real]')
    parser.add_argument('--decoder_type', default='Res2former', type=str, help='reconstruction network type: Unet, RevSCI, ConvFormer, STFormer, Res2former, Res2former-large')
    parser.add_argument('--color_channels', default=1,type=int)
    parser.add_argument('--save_model_step', default=1, type=int)
    parser.add_argument('--save_train_image_step', default=500, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--iter_step', default=400, type=int)
    parser.add_argument('--test_flag', default=False, type=bool)
    parser.add_argument('--train_data_path',type=str,default='DAVIS/JPEGImages/480p', help=' DAVIS 2017 as training dataset, it is available at https://davischallenge.org/davis2017/code.html')
    parser.add_argument('--benchmark_path',type=str,default='testdata/synthetic_data')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--torchcompile',nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")

    args = parser.parse_args()
    return args

