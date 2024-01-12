import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.joint_optimization import DeepOpticsSCI
from schedulers import WarmupStepLR
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import scipy.io as scio
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from opts import parse_args
from test import test
import time
import datetime
from utils import discretize, findzero, checkpoint, TrainData, Logger, save_image, time2file_name, load_checkpoint


def train(args, network, optimizer, pretrain_epoch, logger, weight_path, mask_path, result_path1, result_path2=None, writer=None):
    criterion  = nn.MSELoss()
    criterion = criterion.to(args.device)
    rank = 0
    if args.distributed:
        rank = dist.get_rank()
    dataset = TrainData(args)
    dist_sampler = None
    if args.gray_levels == 2:
        index = torch.arange(0, args.B)[:, None, None].repeat(1, args.size[0], args.size[1]).float()

    if args.distributed:
        dist_sampler = DistributedSampler(dataset, shuffle=True)
        train_data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            sampler=dist_sampler, num_workers=args.num_workers)
    else:
        train_data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    for epoch in range(pretrain_epoch + 1, pretrain_epoch + args.epochs + 1):
        epoch_loss = 0
        network = network.train()
        start_time = time.time()
        for iteration, data in enumerate(train_data_loader):
            gt = data
            gt = gt.float().to(args.device)
            optimizer.zero_grad()
            out_list = network(gt)
            loss = torch.sqrt(criterion(out_list[-1], gt))
            epoch_loss += loss.item()
            loss.backward()
            for p in list(network.parameters()):
                if hasattr(p, 'org'):
                    p.org = p.org.to(gt.device)
                    p.detach().copy_(p.org)
            optimizer.step()
            for p in list(network.parameters()):
                if hasattr(p, 'org'):
                    p.org.copy_(p.detach().clamp_(0,1.0))
                    p.data = discretize(p.org, args.B, args.gray_levels)             

            if rank==0 and (iteration % args.iter_step) == 0:
                encoder_lr = optimizer.state_dict()['param_groups'][0]['lr']
                decoder_lr = optimizer.state_dict()['param_groups'][1]['lr']
                logger.info('epoch: {:<3d}, iter: {:<4d}, loss: {:.5f}, encoder_lr: {:.6f}, decoder_lr: {:.6f}.'
                            .format(epoch, iteration, loss.item(), encoder_lr, decoder_lr))

            if rank==0 and (iteration % args.save_train_image_step) == 0:
                image_out = out_list[-1][0].detach().cpu().numpy()
                image_gt = gt[0].cpu().numpy()
                image_path = './'+ result_path1+ '/'+'epoch_{}_iter_{}.png'.format(epoch, iteration)
                save_image(image_gt, image_out, image_path)

        end_time = time.time()
        if rank==0:
            encoder_lr = optimizer.state_dict()['param_groups'][0]['lr']
            decoder_lr = optimizer.state_dict()['param_groups'][1]['lr']
            logger.info('epoch: {}, avg. loss: {:.5f}, encoder_lr: {:.6f}, decoder_lr: {:.6f}, time: {:.2f}s.\n'
                        .format(epoch, epoch_loss/(iteration+1), encoder_lr, decoder_lr, end_time-start_time))
            writer.add_scalar('Avg. loss - epoch',epoch_loss/(iteration+1),epoch)

        if rank==0 and (epoch % args.save_model_step) == 0:
            model_out_path = './' + weight_path + '/' + 'epoch_{}.pth'.format(epoch)
            if args.distributed:
                checkpoint(epoch, network.module, optimizer, model_out_path)
            else:
                checkpoint(epoch, network, optimizer, model_out_path)
               
            for p in list(network.parameters()):
                if hasattr(p, 'org'):
                    mask_path1 = './' + mask_path + '/' + 'graylevels_{}_Cr_{}_epoch_{}.mat'.format(args.gray_levels, args.B, epoch)
                    mask = p.detach().clone()  
                    if args.train_type == 'real':
                        mask = torch.repeat_interleave(torch.repeat_interleave(mask,2, dim=1), 2, dim=2).float()
                    mask = mask.cpu().numpy()
                    scio.savemat(mask_path1, {'mask': mask})

        if rank==0 and args.test_flag:
            logger.info('epoch: {}, psnr and ssim test results:'.format(epoch))
            if args.distributed:
                psnr_dict, ssim_dict = test(args, network.module, logger, result_path2, writer=writer, epoch=epoch)
            else:
                psnr_dict, ssim_dict = test(args, network, logger, result_path2, writer=writer, epoch=epoch)

            logger.info('psnr_dict: {}.'.format(psnr_dict))
            logger.info('ssim_dict: {}.\n'.format(ssim_dict))



if __name__ == '__main__':
    args = parse_args()
    rank = 0
    pretrain_epoch = 0
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)

    if rank ==0:
        result_path1 = 'results' + '/' + 'E2E-{}-{}graylevels'.format(args.decoder_type, args.gray_levels) + '/' + date_time + '/train'
        weight_path = 'weights' + '/' + 'E2E-{}-{}graylevels'.format(args.decoder_type, args.gray_levels) + '/' + date_time
        mask_path = 'masks' + '/' + 'E2E-{}-{}graylevels'.format(args.decoder_type, args.gray_levels) + '/' + date_time
        log_path = 'log/log' + '/' + 'E2E-{}-{}graylevels'.format(args.decoder_type, args.gray_levels)
        show_path = 'log/show' + '/' + 'E2E-{}-{}graylevels'.format(args.decoder_type, args.gray_levels) + '/' + date_time
        if not os.path.exists(result_path1):
            os.makedirs(result_path1,exist_ok=True)
        if not os.path.exists(weight_path):
            os.makedirs(weight_path,exist_ok=True)
        if not os.path.exists(mask_path):
            os.makedirs(mask_path,exist_ok=True)
        if not os.path.exists(log_path):
            os.makedirs(log_path,exist_ok=True)
        if not os.path.exists(show_path):
            os.makedirs(show_path,exist_ok=True)
        if args.test_flag:
            result_path2 = 'results' + '/' + 'E2E-{}-{}graylevels'.format(args.decoder_type, args.gray_levels) + '/' + date_time + '/test'
            if not os.path.exists(result_path2):
                os.makedirs(result_path2,exist_ok=True)
        else:
            result_path2 = None

    logger = Logger(log_path)
    writer = SummaryWriter(log_dir = show_path)
    
    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device('cuda',local_rank)  ##########
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()

    if rank==0:
        logger.info('\n'+'Date:' + date_time + '\n' +
                'Network Architecture: E2E-{}'.format(args.decoder_type) + '\n' +
                'Simulated or Real: {}'.format(args.train_type) + '\n' +
                'Mask Type: {}_graylevels'.format(args.gray_levels) + '\n' +
                'Image Size: {}'.format(args.size) + '\n' +
                'Compressive Ratio: {}'.format(args.B) + '\n' +
                'Learning Rate: {:.6f}(encoder) \t {:.6f}(decoder)'.format(args.encoder_lr, args.decoder_lr) + '\n' +
                'Train Epochs: {}'.format(args.epochs) + '\n' +
                'Test or Not: {}'.format(args.test_flag) + '\n' +
                'Pretrain Model: {}'.format(args.pretrained_model_path)
                ) 

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    network = DeepOpticsSCI(args).to(args.device)
    optimizer = optim.Adam([{'params': network.encoder.parameters(), 'lr': args.encoder_lr},
                            {'params': network.decoder.parameters()}], lr=args.decoder_lr)

    if rank==0:
        if args.pretrained_model_path is not None:
            logger.info('Loading pretrained model...')
            pretrained_dict = torch.load(args.pretrained_model_path)
            if 'pretrain_epoch' in pretrained_dict.keys():
                pretrain_epoch = pretrained_dict['pretrain_epoch']
                logger.info('Pretrain epoch: {}'.format(pretrain_epoch))      
            load_checkpoint(network, pretrained_dict,optimizer)
        else:
            logger.info('No pretrained model.')

    if args.distributed:
        network = DDP(network, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    train(args, network, optimizer, pretrain_epoch, logger, weight_path, mask_path, result_path1, result_path2, writer)
    writer.close()


