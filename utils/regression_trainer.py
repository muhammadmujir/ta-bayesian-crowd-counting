from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from  models.vgg import vgg19
# from models.custom_model import CSRNet
from datasets.crowd import Crowd
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob
import shutil

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'model_best.pth.tar'))  

resultCSV = None
resultPath = None

class RegTrainer(Trainer):

    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        global resultPath, resultCSV
        args = self.args
        resultPath = args.save_dir
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            self.device = torch.device("cpu")
            self.device_count = 1
            # raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}
        self.model = vgg19()
        # self.model = CSRNet()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        # if args.resume:
        #     suf = args.resume.rsplit('.', 1)[-1]
        #     if suf == 'tar':
        #         checkpoint = torch.load(args.resume, self.device)
        #         self.model.load_state_dict(checkpoint['model_state_dict'])
        #         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #         self.start_epoch = checkpoint['epoch'] + 1
        #     elif suf == 'pth':
        #         self.model.load_state_dict(torch.load(args.resume, self.device))
        
        self.best_prec = 1e8
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                if torch.cuda.is_available():
                    checkpoint = torch.load(args.resume)
                else:
                    checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
                self.start_epoch = checkpoint['epoch']
                self.best_prec = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resyne))
                
        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0

    def train(self):
        """training process"""
        global resultPath, resultCSV
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            
            if (epoch == 0):
                resultCSV = open(os.path.join(resultPath, 'result.csv'), 'w')
                resultCSV.write('%s;' % "Epoch")
                resultCSV.write('%s;' % "Training Loss (BAYESIAN)")
                resultCSV.write('%s;' % "Training MAE")
                resultCSV.write('%s;' % "Training MSE")
                resultCSV.write('%s;' % "Testing MAE BY COUNT")
                resultCSV.write('%s;' % "Testing MSE BY COUNT")
                resultCSV.write('\n')
            else:
                resultCSV = open(os.path.join(resultPath, 'result.csv'), 'a')
            
            # logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            
            self.epoch = epoch
            self.train_eopch(epoch)
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                prec = self.val_epoch()
                is_best = prec < self.best_prec
                self.best_prec = min(prec, self.best_prec)
                print(' * best MAE {mae:.3f} '.format(mae=self.best_prec))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.resume,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': self.best_prec,
                    'optimizer' : self.optimizer.state_dict(),
                }, is_best, resultPath)
            resultCSV.write('\n')

    def train_eopch(self, epoch):
        global resultCSV
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        # epoch_start = time.time()
        self.model.train()  # Set model to training mode
        end = time.time()
        
        resultCSV.write('%s;' % "Epoch: {}".format(epoch))
    
        # Iterate over data.
        for step, (inputs, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
            data_time.update(time.time() - end)
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                prob_list = self.post_prob(points, st_sizes)
                loss = self.criterion(prob_list, targets, outputs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)
            
            batch_time.update(time.time() - end)
            end = time.time()
        
            if step % self.args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      .format(
                       epoch, step, len(self.dataloaders['train']), batch_time=batch_time,
                       data_time=data_time, loss=epoch_loss))
            
        resultCSV.write('%s;' % str(epoch_loss.avg).replace(".", ",",1))
        resultCSV.write('%s;' % str(epoch_mae.avg).replace(".", ",",1))
        resultCSV.write('%s;' % str(np.sqrt(epoch_mse.avg)).replace(".", ",",1))
        # logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
        #              .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
        #                      time.time()-epoch_start))
        
        # model_state_dic = self.model.state_dict()
        # save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        # torch.save({
        #     'epoch': self.epoch,
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     'model_state_dict': model_state_dic
        # }, save_path)
        # self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        print("---------begin test----------")
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        # Iterate over data.
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        print(' * MAE BY COUNT {mae:.3f} '.format(mae=mae))
        print(' * MSE BY COUNT {mse:.3f} '.format(mse=mse))
        resultCSV.write('%s;' % str(mae).replace(".", ",",1))
        resultCSV.write('%s;' % str(mse).replace(".", ",",1))
        return mae
        # logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
        #              .format(self.epoch, mse, mae, time.time()-epoch_start))

        # model_state_dic = self.model.state_dict()
        # if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
        #     self.best_mse = mse
        #     self.best_mae = mae
        #     logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
        #                                                                          self.best_mae,
        #                                                                          self.epoch))
        #     torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))



