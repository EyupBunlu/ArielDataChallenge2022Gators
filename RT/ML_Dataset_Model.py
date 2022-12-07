from torch import nn
from torch.utils.data import Dataset
import torch

class Parameter_ML(nn.Module):
    def __init__(self,Dropout=0):
        super(Parameter_ML, self).__init__()
    
        self.attention_layer = nn.Sequential( nn.Linear(in_features=52, out_features = 64),nn.ReLU(),nn.Linear(in_features=64, out_features = 64),nn.ReLU(),nn.Linear(in_features=64, out_features = 52))
        
        self.aux_layer = nn.Sequential( nn.Linear(in_features=12, out_features = 64),nn.ReLU(),nn.Linear(in_features=64, out_features = 64),nn.ReLU(), nn.Linear(in_features=64, out_features = 64))
        
        
        self.main_layer = nn.Sequential( nn.Linear(in_features=116, out_features = 192), nn.ReLU(), nn.Linear(in_features=192, out_features = 384), nn.ReLU(), nn.Linear(in_features=384, out_features = 384),nn.Dropout(Dropout),nn.ReLU() ,nn.Linear(in_features=384, out_features = 20),nn.Sigmoid()) 

        self.noise_layer = nn.Sequential( nn.Linear(in_features=116, out_features = 64),nn.ReLU(),nn.Linear(in_features=64, out_features = 64),nn.ReLU(),nn.Linear(in_features=64, out_features = 52)) 

    def forward(self, input1, input2,input3): #input1:spectra, input2: auxxilary, input3:noise


        a = input1*self.noise_layer(torch.cat((input3,input2,input1),axis=1 ) )
        b=torch.cat( (self.attention_layer(a)*a,self.aux_layer(input2) ) ,axis=1 )
        return self.main_layer(b).reshape(a.shape[0],4,5) *torch.Tensor([1,11,11,6])[None,:,None]-torch.Tensor([0,12,12,0])[None,:,None]




 
class Parameter_SML(nn.Module):
    def __init__(self,Dropout=0): # ,quart_mn,quart_std,
        super(Parameter_SML, self).__init__()
    
        self.attention_layer = nn.Sequential( nn.Linear(in_features=52, out_features = 64),nn.ReLU(),nn.Linear(in_features=64, out_features = 64),nn.ReLU(),nn.Linear(in_features=64, out_features = 52))
        
        self.aux_layer = nn.Sequential( nn.Linear(in_features=12, out_features = 64),nn.ReLU(),nn.Linear(in_features=64, out_features = 64),nn.ReLU(), nn.Linear(in_features=64, out_features = 64),nn.ReLU(),nn.Linear(in_features=64, out_features = 64))
        
        
        self.main_layer = nn.Sequential( nn.Linear(in_features=168, out_features = 192), nn.ReLU(), nn.Linear(in_features=192, out_features = 384), nn.ReLU(), nn.Linear(in_features=384, out_features = 384),nn.Dropout(Dropout),nn.ReLU() ,nn.Linear(in_features=384, out_features = 20),
                                       nn.Sigmoid()) 
        # self.quart_mn = quart_mn
        # self.quart_std = quart_std

        self.noise_layer = nn.Sequential( nn.Linear(in_features=12, out_features = 64),nn.ReLU(),nn.Linear(in_features=64, out_features = 64),nn.ReLU(),nn.Linear(in_features=64, out_features = 1)) 

    def forward(self, input1, input2,input3): # ,give_constant=False #input1:spectra, input2: auxxilary, input3:noise

        enhance = self.noise_layer(input2)


        b=torch.cat( (enhance*input1,input3*enhance,self.aux_layer(input2) ) ,axis=1 )
        return self.main_layer(b).reshape(input1.shape[0],4,5)*torch.Tensor([1,11,11,6])[None,:,None]-torch.Tensor([0,12,12,0])[None,:,None]

    
    
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
        sample={'input1': self.data[idx], 'output': self.labels[idx],'input3':self.noise[idx,:],'input2':self.aux[idx]}
        if self.transform is not None:          
            sample = self.transform(sample)
        return sample










