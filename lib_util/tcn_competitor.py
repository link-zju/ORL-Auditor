import os
import torch
import torch.nn.functional as F
import pickle
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from lib_util.tcn import TemporalConvNet

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.tcn(x)
        x = self.linear(x[:, :, -1])
        return F.log_softmax(x, dim=1)
    


class TCNCompetitor:
    @staticmethod
    def tcn_model_generate(dataset_save_path, cuda_id, train_epoch=100, learning_rate=1e-3, dropout=0.2):
        # load buffer
        device = torch.device('cuda', cuda_id)        

        with open(dataset_save_path, "rb") as handle:
            dataset_dict = pickle.load(handle)
        print(dataset_dict.keys)
        
        X_train, X_test, y_train, y_test = train_test_split(dataset_dict['x'], dataset_dict['y'], test_size=0.2, random_state=42)
        
        X_train = np.transpose(X_train, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))
        
        X_train = torch.Tensor(X_train)
        X_test = torch.Tensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2)

        input_channel = X_train.shape[1]
        level = 8
        num_channels = [input_channel] * level
        output_size = 2 ## two categories: positive or negative
        kernel_size = 7
        dropout = 0.2
        
        tcn_model = TCN(input_channel, output_size, num_channels, kernel_size, dropout).to(device)
        optimizer = torch.optim.Adam(tcn_model.parameters(), lr=learning_rate)
        
        for i in range(train_epoch):
            train_loss = 0.0
            train_acc = 0
            for _, data in enumerate(train_loader, 0):
                x, y = data[0].to(device), data[1].to(device)
                
                y_pred = tcn_model(x)
                
                train_acc+=sum(torch.argmax(y_pred, dim=1)==y)              
                loss = F.nll_loss(y_pred, y)
                
                optimizer.zero_grad()  
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            
            train_acc = train_acc/len(train_loader.dataset) * 100
            print(f'[epoch: {i}] train loss: {train_loss:.3f} train acc:  {train_acc:.2f}')  
            
            if i%10 == 9:
                ## test loss
                with torch.no_grad():
                    test_loss = 0.0
                    test_acc = 0.0
                    for _, data in enumerate(test_loader, 0):
                        x, y = data[0].to(device), data[1].to(device)
                        
                        y_pred = tcn_model(x)
                        loss = F.nll_loss(y_pred, y)
                        
                        test_acc+=sum(torch.argmax(y_pred, dim=1)==y)
                        
                        test_loss += loss.item()
                
                test_acc = test_acc/len(test_loader.dataset) * 100
                print(f'[epoch: {i}] test loss: {test_loss:.3f} test acc:  {test_acc:.2f}')    
                
                
                model_save_path = dataset_save_path.replace('competitor_classifier_dataset', 'competitor_classifier_model')
                model_save_path = model_save_path[:model_save_path.find(".")]
                
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)    
                
                path = os.path.join(model_save_path, 'ckpt_{}.pt'.format(i+1))         
                torch.save({'epoch': i,
                            'model_state_dict': tcn_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss,
                            'test_loss': test_loss,
                            'train_acc': train_acc,
                            'test_acc': test_acc
                            },  path)
                                
        print('Finished Training')

        
    