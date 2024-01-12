import torch 
from torch import nn 
from models.physical_layer import encoder
from models.digital_layer import STFormer, ConvFormer, Unet, RevSCI, Res2former

class DeepOpticsSCI(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.encoder = encoder(args)
        if args.decoder_type == 'Unet':
            self.decoder = Unet(in_ch=args.B, out_ch=64)
        elif args.decoder_type == 'RevSCI':
            self.decoder = RevSCI(num_block=50)
        elif args.decoder_type == 'ConvFormer':
            self.decoder = ConvFormer(num_block=8, num_group=2)
        elif args.decoder_type == 'STFormer':
            self.decoder = STFormer(color_channels=args.color_channels,units=4,dim=64,frames=args.B)
        elif args.decoder_type == 'Res2former':
            self.decoder = Res2former(dim=96,stage_num=1,depth_num=[3,3],color_ch=args.color_channels)
        elif args.decoder_type == 'Res2former_large':
            self.decoder = Res2former(dim=128,stage_num=1,depth_num=[5,5],color_ch=args.color_channels)
        else:
            raise TypeError('Reconstruction method undefined!')
        

    def forward(self, gt_batch):
        meas_batch, mask = self.encoder(gt_batch)         
        rec_batch = self.decoder(meas_batch, mask)

        return rec_batch

    