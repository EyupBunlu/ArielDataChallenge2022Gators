from torch import nn
from torch.utils.data import Dataset
import torch

# Contains the model and Dataset
    
    
#########################################################################################
class Combined_QuartM_ML(nn.Module):
    def __init__(self,quart_mn,quart_std,Dropout=0):
        super(Combined_QuartM_ML, self).__init__()
    
        self.attention_layer = nn.Sequential( nn.Linear(in_features=52, out_features = 64),nn.ReLU(),
                                             nn.Linear(in_features=64, out_features = 64),nn.ReLU(),
                                             nn.Linear(in_features=64, out_features = 52))
        
        self.aux_layer = nn.Sequential( nn.Linear(in_features=12, out_features = 64),nn.ReLU(),
                                       nn.Linear(in_features=64, out_features = 64),nn.ReLU(),
                                       nn.Linear(in_features=64, out_features = 64))
        
        
        self.main_layer = nn.Sequential( nn.Linear(in_features=116, out_features = 192), nn.ReLU(),
                                        nn.Linear(in_features=192, out_features = 384), nn.ReLU(),
                                        nn.Linear(in_features=384, out_features = 384),nn.Dropout(Dropout),
                                        nn.ReLU() ,nn.Linear(in_features=384, out_features = 15)) 
        self.quart_mn = quart_mn
        self.quart_std = quart_std

        self.scale_layer = nn.Sequential( nn.Linear(in_features=116, out_features = 64),nn.ReLU(),
                                         nn.Linear(in_features=64, out_features = 64),nn.ReLU(),
                                         nn.Linear(in_features=64, out_features = 52)) 

    def forward(self, input1, input2,input3): #input1:spectra, input2: auxxilary, input3:noise


        a = input1*self.scale_layer(torch.cat((input3,input2,input1),axis=1 ) )
        b=torch.cat( (self.attention_layer(a)*a,self.aux_layer(input2) ) ,axis=1 )
        return self.main_layer(b).reshape(a.shape[0],self.quart_mn.shape[0],self.quart_mn.shape[1])*self.quart_std+self.quart_mn 
    
    
    
class Combined_Quart_Dataset(Dataset):
    def __init__(self,spectra,aux,quart,noise=None,transform=None):
        self.data = spectra
        self.labels = quart
        self.noise = noise
        self.aux = aux
        self.transform = transform

    def __len__(self):

        return len(self.labels)
    def __getitem__(self, idx):
        sample={'input1': self.data[idx,:], 'output': self.labels[idx],'input2':self.aux[idx,:], 'input3':self.noise[idx,:] }

        if self.transform is not None:          
            sample = self.transform(sample)
        return sample


  




















