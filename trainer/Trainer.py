import torch
import time
import copy
import random
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.optim.swa_utils import AveragedModel, SWALR


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, default_tau, batch_size=128, num_epochs=250, weight_decay=0.001):

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        #self.lr = (batch_size/2048)**(1/2)/1000
        self.lr = 0.00005
        self.weight_decay = weight_decay
        self.num_gpus = torch.cuda.device_count()

        self.tau_list = [0.25, 0.375, 0.5, 0.625]
        # self.tau_list = [0.375, 0.5, 0.625, 0.75]     
        self.default_tau = default_tau

        self.train_time = 0.
        self.test_time = 0.

        self.train_acc = 0
        self.test_acc = 0        
        self.train_acc_list = []
        self.test_acc_list = []

        self.acc_dict = dict()

        # Data
        print('\n==> Preparing data..')
        self.trainloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       num_workers=4*self.num_gpus,
                                                       pin_memory=True)
        self.testloader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       num_workers=4*self.num_gpus,
                                                       pin_memory=True)

        self.transform_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
        ])

        self.classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']


        # Model
        print('\n==> Building model..')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model = self.model.to(self.device)
        self.swa_model = AveragedModel(self.model, device=self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        #self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=5e-5)
        self.scaler = torch.cuda.amp.GradScaler()


    def update_params(self):
        for p_model, p_swa in zip(self.model.parameters(), self.swa_model.parameters()):
            p_model.detach().copy_(p_swa.detach().to(self.device))
        
        b_model_list = []
        b_swa_list = []

        for b_model, b_swa in zip(self.model.buffers(), self.swa_model.buffers()):
            b_model_list.append(b_model)
            b_swa_list.append(b_swa)
        
        del b_swa_list[0]

        for b_model, b_swa in zip(b_model_list, b_swa_list):
            b_model.detach().copy_(b_swa.detach().to(self.device))


    def update(self):
        # self.swa_model.update_parameters(self.model)
        # torch.optim.swa_utils.update_bn(self.trainloader, self.swa_model, device=self.device)
        # self.update_params()
        return 0


    def print_acc(self):
        print('\n==> Printing test acc..')
        for i in list(self.acc_dict.items()):
            print(f"Epoch: {i[0]} | Test Acc: {i[1]:.2f}%")
        
        self.train_acc_list = [round(x, 2) for x in self.train_acc_list]
        self.test_acc_list = [round(x, 2) for x in self.test_acc_list]

        print(f'\nTrain acc list: {self.train_acc_list}')
        print(f'\nTest acc list: {self.test_acc_list}')


    def train(self, swa_with_tau=False):
        if swa_with_tau:
            print('\n==> Training model swa with tau..')    
        else:
            print('\n==> Training model normal..')

        for epoch in range(1, self.num_epochs+1):

            train_start = time.time()

            if epoch < (self.num_epochs * 0.1) :
                self.train_one_epoch_normal(epoch)
            else:
                if (epoch % 5) == 0 and not swa_with_tau:
                    self.train_one_epoch_normal(epoch)
                    self.update()
                    self.swa_scheduler.step()
                elif (epoch % 5) == 0 and swa_with_tau:
                    self.train_one_epoch_swa_with_tau(epoch)
                    self.swa_scheduler.step()
                else:
                    self.train_one_epoch_normal(epoch)

            train_end = time.time()
            self.train_time += (train_end-train_start)/60
            print(f"Epoch: [{epoch}/{self.num_epochs}] | Train time elapse: {self.train_time:.2f} min")
            
            # scheduler update
            #self.scheduler.step()

            self.train_acc_list.append(self.train_acc)
            self.test()
            self.test_acc_list.append(self.test_acc)
            if (epoch % 10) == 0:
                self.acc_dict[epoch] = self.test_acc

        self.update()


    # Training
    def train_one_epoch_normal(self, epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (samples, targets) in enumerate(self.trainloader):
            samples = samples.to(self.device, memory_format=torch.channels_last, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # 0.4 확률로 transform
            r = random.random()
            if r < 0.4:
                samples = self.transform_augmentation(samples)

            samples = torch.cat((samples,samples),dim=0)
            samples = torch.split(samples, samples.shape[0]//2, dim=0)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs1 = self.model(samples[0], train_mode=True, tau=self.default_tau)
                outputs2 = self.model(samples[1], train_mode=True, tau=self.default_tau)
                # from git
                loss = 0.25 * self.criterion(outputs1, targets)
                loss = loss + 0.25 * self.criterion(outputs2, targets)
                loss = loss + 0.25 * self.criterion(outputs1, outputs2.detach().sigmoid())
                loss = loss + 0.25 * self.criterion(outputs2, outputs1.detach().sigmoid())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_value = loss.item()
            train_loss += loss_value
            _, predicted = outputs1.max(1) 
            _, targets = targets.max(1)              
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()   

            if (batch_idx+1) % 90 == 0:
                print(f"Epoch: [{epoch}/{self.num_epochs}] | Batch: [{batch_idx+1}/{len(self.trainloader)}] | Loss: {train_loss/(batch_idx+1):.4f} | Train Acc: {100.*correct/total:.2f}% ({correct}/{total})")
        
        self.train_acc = 100 * (correct / total)


    def train_one_epoch_swa_with_tau(self, epoch):
        def process_samples(samples, targets, tau):
            
            samples = torch.cat((samples, samples), dim=0)
            # if tau != self.default_tau:
            #     samples = self.transform_augmentation(samples)
            samples = self.transform_augmentation(samples)
            samples = torch.split(samples, samples.shape[0]//2, dim=0)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs1 = self.model(samples[0], train_mode=True, tau=tau)
                outputs2 = self.model(samples[1], train_mode=True, tau=tau)
                loss = 0.25 * self.criterion(outputs1, targets)
                loss = loss + 0.25 * self.criterion(outputs2, targets)
                loss = loss + 0.25 * self.criterion(outputs1, outputs2.detach().sigmoid())
                loss = loss + 0.25 * self.criterion(outputs2, outputs1.detach().sigmoid())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            return outputs1, loss

        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        correct_ = 0
        total_ = 0

        for batch_idx, (samples, targets) in enumerate(self.trainloader):
            samples = samples.to(self.device, memory_format=torch.channels_last, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs, loss = process_samples(samples, targets, self.default_tau)
            
            loss_value = loss.item()
            train_loss += loss_value
            _, predicted = outputs.max(1)
            _, targets = targets.max(1)              
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()   

            total_ += targets.size(0)
            correct_ += predicted.eq(targets).sum().item()   

            if (batch_idx + 1) % 90 == 0:
                print(f"Epoch: [{epoch}/{self.num_epochs}] | Tau: {self.default_tau} | Batch: [{batch_idx+1}/{len(self.trainloader)}] | Loss: {train_loss / (batch_idx+1):.4f} | Train Acc: {100. * correct / total:.2f}% ({correct}/{total})")

        self.update()

        for tau in self.tau_list:
            if tau != self.default_tau:

                train_loss = 0
                correct = 0
                total = 0

                for batch_idx, (samples, targets) in enumerate(self.trainloader):
                    samples = samples.to(self.device, memory_format=torch.channels_last, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    outputs, loss = process_samples(samples, targets, tau)

                    loss_value = loss.item()
                    train_loss += loss_value
                    _, predicted = outputs.max(1)
                    _, targets = targets.max(1)                               
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()   

                    total_ += targets.size(0)
                    correct_ += predicted.eq(targets).sum().item()   

                    if (batch_idx + 1) % 90 == 0:
                        print(f"Epoch: [{epoch}/{self.num_epochs}] | Tau: {tau} | Batch: [{batch_idx+1}/{len(self.trainloader)}] | Loss: {train_loss / (batch_idx+1):.4f} | Train Acc: {100. * correct / total:.2f}% ({correct}/{total})")

                self.update()

        self.train_acc = 100 * (correct_ / total_)

    def test(self):

        print('\n==> Testing normal model..')
        
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (samples, targets) in enumerate(self.testloader):
                samples = samples.to(self.device, memory_format=torch.channels_last, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                predictions = self.model(samples)

                loss = self.criterion(predictions, targets).detach()

                test_loss += loss.item()
                _, predicted = predictions.max(1)
                _, targets = targets.max(1)              
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # if (batch_idx + 1) % 10 == 0:
                #     print(f"Batch: [{batch_idx+1}/{len(self.testloader)}] | Loss: {test_loss / (batch_idx+1):.4f} | Test Acc: {100. * correct / total:.2f}% ({correct}/{total})")

            self.test_acc = 100 * (correct / total)
            print(f"Final Test Acc: {100. * correct / total:.2f}% ({correct}/{total})")
