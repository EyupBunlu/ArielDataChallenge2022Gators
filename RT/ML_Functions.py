import os
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from tqdm.notebook import tqdm
from time import time
import numpy as np  
import scipy
import h5py

# Contains training function, preprocessor, distribution functions and some small functions 
def standardize(data,mean,std):
    return (data-mean)/std


# Create the CDF of the distribution over x
def dist_func(params,x):
    temp = params[0]*1/2*(1+scipy.special.erf( (x-params[2])/(params[3]*np.sqrt(2) ) )  )
    temp[x>params[1]] += 1-params[0]
    temp[x<=params[1]] += (1-params[0])*(-12-x[x<=params[1]])/(-12-params[1])
    return temp

# Sample from the distribution
def sample_dist(params,N=4998):
    dist = np.zeros([N,params.shape[0],params.shape[-1]])
    x = np.linspace(-12,-1,10000)
    
    
    for i in tqdm(range(params.shape[0])):
        
        for j in range(5):
            sample = dist_func(params[i,:,j],x)
            sample[0]=0
            sample[-1]=1
            for k in range(N):
                r = np.random.uniform()

                dist[k,i,j]= x[np.abs(r-sample).argmin()]     
              
    return dist

# Parametrize tracedata to feed it into the ML algorithm
def convert_dist(quart_tr,trace_path):
    f = h5py.File(trace_path,'r')
    new_params = np.zeros((quart_tr.shape[0],4,5))
    def dist_diff(params,x,y):
        temp = (1-params[0])*1/2*(1+scipy.special.erf( (x-params[2])/(params[3]*np.sqrt(2) ) )  )
        temp[x>params[1]] += params[0]
        temp[x<=params[1]] += params[0]*(-12-x[x<=params[1]])/(-12-params[1])
        return y-temp

    for k in tqdm(range(quart_tr.shape[0])):

    
        temp = np.array(f['Planet_'+str(k)]['tracedata'][:])
        W   = np.array(f['Planet_'+str(k)]['weights'][:])

        X = [[],]
        Y = [[],]
        for i in range(1,6):

            x = np.linspace(temp[:,i].min(),temp[:,i].max(),1000)
            if (-temp[:,i].min()+temp[:,i].max())<1:
                x = np.linspace(temp[:,i].min(),temp[:,i].max(),100)
            y = np.zeros(x.shape)
            for re in range(x.shape[0]):
                y[re] = ( W[temp[:,i]<=x[re]]).sum()

            X.append(x)
            Y.append(y)
            
            new_params[k,:,i-1] = scipy.optimize.least_squares(dist_diff
            ,np.array([0.01,quart_tr[k,0,i],quart_tr[k,1,i],(quart_tr[k,2,i]-quart_tr[k,0,i])/2]),args=(X[i],Y[i]),
                                             bounds = ([0,-12,-12,-0],[1,-1,-1,6]) ).x
    return new_params


def train(model,optimizer, loss_fn, train_dl, val_dl, epochs,lrs, device):
    
    history = {} 

    history['val_loss'] = []
    history['loss'] = []
    history['worst_of'] = []
    
    start_time_sec = time()

    bars = [tqdm(range( i),leave=True) for i in epochs]

    for lr,bar in zip(lrs,bars):
        for g in optimizer.param_groups:
            g['lr'] = lr
        for epoch in bar:

            model.train()
            train_loss         = 0.0
            num_train_examples = 0

            for batch in tqdm(train_dl,leave=False):

                optimizer.zero_grad()

                y = batch.pop('output')
                yhat = model(**batch)

                loss = loss_fn(y,yhat).mean()
                loss.backward()

                optimizer.step()

                train_loss         += loss.data.item() * yhat.shape[0]
                num_train_examples += yhat.shape[0]

            train_loss  = train_loss / len(train_dl.dataset)



            # --- EVALUATE ON VALIDATION SET -------------------------------------
            model.eval()
            val_loss       = 0.0
            worst_of = 0.0
            num_val_examples = 0

            for batch in val_dl:


                y = batch.pop('output')
                yhat = model(**batch)

                loss = loss_fn(y,yhat).detach()
                val_loss         += loss.mean().item()* yhat.shape[0]


            val_loss = val_loss / len(val_dl.dataset)
            worst_of = worst_of / len(val_dl.dataset)


            history['worst_of'].append(100*np.sqrt(worst_of))
            history['loss'].append(100*np.sqrt(train_loss))
            history['val_loss'].append(100*np.sqrt(val_loss))

            if np.min(history['val_loss'])== history['val_loss'][-1]:
                state = model.state_dict().copy()

            text = [ 'Tr: %.2f, Val: %.2f, W: %.2f, lr: %.1E'%(100*np.sqrt(train_loss),100*np.sqrt(val_loss),100*np.sqrt(worst_of),optimizer.param_groups[0]['lr'])]

            bar.set_postfix_str(*text,refresh=True)

    end_time_sec       = time()
    total_time_sec     = end_time_sec - start_time_sec
    
    print('Time total: %5.2f sec' % (total_time_sec) + ', Best Val. Loss:{}'.format(np.min(history['val_loss']))+', Best of Worst of:{}'.format(np.min(history['worst_of'])))
    
    return {'state':state,'history':history}




def show_model(model):
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        
def preprocessor(spec,aux,noise,ind=None,std_mn=None):

    # Modifying outliers
    for i in range(spec.shape[0]):
        
        spec[i,spec[i,:]>0.1] = (spec[i,spec[i,:]<0.1]/noise[i,spec[i,:]<0.1]).mean()*noise[i,spec[i,:]>0.1]
    
    # Calculate R_p/R_s for the next steps
    temp = ((aux[:,7]/aux[:,2]))[:,None]
    
    # Calculating the effective temperature and the scale height
    T = aux[:,3]*(1/2*aux[:,2]/(aux[:,6]*149597870.7*1000))**(1/2)
    H = 1.380649* T / (2.29*1.66054e-4*aux[:,8])
    
    # Getting rid of the bulk planet contribution
    spec = (spec - temp**2)
    
    # Concatenate three unitless features to the auxillary data
    aux = torch.cat((aux,(aux[:,7]/aux[:,2])[:,None],
                        (aux[:,6]/H)[:,None],spec.max(axis=1).values[:,None] *aux[:,2][:,None]/H[:,None]  ),axis=1) # *

    # Normalization of noise and spectra
    noise = noise/spec.max(axis=1).values[:,None]
    spec = spec/spec.max(axis=1).values[:,None]

    # Calculate std and mean for the auxillary data if it is the training phase
    if std_mn is None:
        aux_std,aux_mn = aux[ind].std(axis=0),aux[ind].mean(axis=0)
        std_mn = [aux_std,aux_mn]
    
    else:
        aux_std,aux_mn = std_mn
       
    # Standardize the auxillary data
    aux = standardize(aux,aux_mn,aux_std) 
    
    return spec,aux,noise,T,std_mn
