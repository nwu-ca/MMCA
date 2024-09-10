import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.autograd import Variable
import numpy as np
import pickle
import os
from tensorboardX import SummaryWriter
import time
# import metrics

class ModelNetTrainer(object):

    def __init__(self, model, train_loader, val_loader, optimizer, \
                 model_name, log_dir, num_views=12):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        # self.loss_fn = get_loss()
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views

       
        self.model.cuda()
        if self.log_dir is not None:
            
            self.writer = SummaryWriter(log_dir)

    def get_loss(self, net_output, ground_truth):
        
        gender_loss = F.cross_entropy(net_output['gender'], ground_truth['gender_label'])
        ethnic_loss = F.cross_entropy(net_output['ethnic'], ground_truth['ethnic_label'])
        loss = 0.7 * gender_loss + 0.3 * ethnic_loss
        return loss, {'gender': gender_loss, 'ethnic': ethnic_loss}

    def calculate_metrics(self,output, target):
        _, predicted_gender = output['gender'].cpu().max(1)
        gt_gender = target['gender_label'].cpu()

        _, predicted_ethnic = output['ethnic'].cpu().max(1)
        gt_ethnic = target['ethnic_label'].cpu()

        with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
            warnings.simplefilter("ignore")
            accuracy_gender = balanced_accuracy_score(y_true=gt_gender.numpy(), y_pred=predicted_gender.numpy())
            accuracy_ethnic = balanced_accuracy_score(y_true=gt_ethnic.numpy(), y_pred=predicted_ethnic.numpy())
        return accuracy_gender, accuracy_ethnic

    def get_model_memory_usage(self, input_size):

        inputs = torch.randn(input_size)
        total_params = sum(p.numel() for p in self.model.parameters())
        total_bytes = total_params * 4  
        total_megabytes = total_bytes / (1024 * 1024)
        return total_megabytes

    def train_model_and_measure_time(self, n_epochs):
        
        start_time = time.time()
        self.train(n_epochs)
        end_time = time.time()
        training_time_seconds = end_time - start_time
        training_time_minutes = training_time_seconds / 60
        return training_time_minutes

    def train(self, n_epochs):

        best_acc = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.train()
        for epoch in range(n_epochs):
            start_memory = torch.cuda.memory_allocated(device)
           
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths)/self.num_views))
            filepaths_new = [] 
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            
            self.writer.add_scalar('params/lr', lr, epoch)

            # train one epoch
            out_data = None
            in_data = None
            total_loss = 0
            accuracy_gender = 0
            accuracy_ethnic = 0
            for i1, data in enumerate(self.train_loader):
                
                if self.model_name == 'mvcnn':
                   
                    N,V,C,H,W = data[1].size()
                    in_data = Variable(data[1]).view(-1,C,H,W).cuda()
                   
                else:
                    in_data = Variable(data[1].cuda())
                    
                target_labels = data[0] # dict
                # print(target_labels)
                target = {t: target_labels[t][0].to(device) for t in target_labels}  #Variable().cuda().to(torch.float)

                self.optimizer.zero_grad()
                
                out_data = self.model(in_data)

                
                loss_train, losses_train = self.get_loss(out_data, target)
               
                total_loss += loss_train
                batch_accuracy_gender, batch_accuracy_ethnic = \
                    self.calculate_metrics(out_data, target)
                accuracy_gender += batch_accuracy_gender
                accuracy_ethnic += batch_accuracy_ethnic

                loss_train.backward()
                self.optimizer.step()#参数更新

                
            end_memory = torch.cuda.memory_allocated(device)
            print(f'Epoch {epoch}: Start Memory: {start_memory} bytes, End Memory: {end_memory} bytes')
            print("tarin/epoch {:4d}, loss: {:.4f}, gender: {:.4f}, ethnic: {:.4f}".format(
                epoch,
                total_loss / len(self.train_loader),
                accuracy_gender / len(self.train_loader),
                accuracy_ethnic / len(self.train_loader)
                ))
            self.writer.add_scalar('train_loss', total_loss/len(self.train_loader), epoch + 1)
            self.writer.add_scalar('gender_acc:', accuracy_gender / len(self.train_loader), epoch + 1)
            self.writer.add_scalar('ethnic_acc:', accuracy_ethnic / len(self.train_loader), epoch + 1)
            
            # evaluation
            if (epoch+1)%1==0: 
                with torch.no_grad():
                    val_loss, val_gender, val_ethnic = self.update_validation_accuracy(epoch)
                val_acc = (val_gender + val_ethnic)/2
                self.writer.add_scalar('val/val_loss', val_loss, epoch+1)
                self.writer.add_scalar('val/sex', val_gender, epoch + 1)
                self.writer.add_scalar('val/ancestry', val_ethnic, epoch + 1)
                


            # save best model
            if val_acc > best_acc:
                best_acc = val_acc
                self.model.save(self.log_dir, epoch)
 
            # adjust learning rate manually
            
            if epoch > 0 and (epoch+1) % 10 == 0:
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.5

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir+"/all_scalars.json")
        self.writer.close()

    def update_validation_accuracy(self, epoch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_loss = 0
        
        self.model.eval()
        accuracy_gender = 0
        accuracy_ethnic = 0
        for _, data in enumerate(self.val_loader, 0):

            if self.model_name == 'mvcnn':
                N,V,C,H,W = data[1].size()
                in_data = Variable(data[1]).view(-1,C,H,W).cuda()
            else:#'svcnn'
                in_data = Variable(data[1]).cuda()

            # target = Variable(data[0]).cuda()
            target_labels = data[0]  # dict
            target = {t: target_labels[t][0].to(device) for t in target_labels}  # Variable().cuda().to(torch.float)

            out_data = self.model(in_data)

            loss_validate, loss_validates = self.get_loss(out_data, target)
            all_loss += loss_validate
            # validate
            batch_accuracy_gender, batch_accuracy_ethnic = \
                self.calculate_metrics(out_data, target)
            accuracy_gender += batch_accuracy_gender
            accuracy_ethnic += batch_accuracy_ethnic

        print("val/epoch {:4d}, loss: {:.4f}, gender: {:.4f}, ethnic: {:.4f}".format(
            epoch,
            all_loss / len(self.val_loader),
            accuracy_gender / len(self.val_loader),
            accuracy_ethnic / len(self.val_loader)
            ))
        self.model.train()

        return all_loss / len(self.val_loader), accuracy_gender / len(self.val_loader), accuracy_ethnic / len(self.val_loader)


