import numpy as np
import h5py
from tqdm.notebook import tqdm
import pandas as pd

def to_observed_matrix(data_file,aux_file):
    # careful, orders in data files are scambled. We need to "align them with id from aux file"
    num = len(data_file.keys())
    id_order = aux_file['planet_ID'].to_numpy()
    observed_spectrum = np.zeros((num,52,4))

    for idx, x in enumerate(id_order):
        current_planet_id = f'Planet_{x}'
        instrument_wlgrid = data_file[current_planet_id]['instrument_wlgrid'][:]
        instrument_spectrum = data_file[current_planet_id]['instrument_spectrum'][:]
        instrument_noise = data_file[current_planet_id]['instrument_noise'][:]
        instrument_wlwidth = data_file[current_planet_id]['instrument_width'][:]
        observed_spectrum[idx,:,:] = np.concatenate([instrument_wlgrid[...,np.newaxis],
                                            instrument_spectrum[...,np.newaxis],
                                            instrument_noise[...,np.newaxis],
                                            instrument_wlwidth[...,np.newaxis]],axis=-1)
    return observed_spectrum


def standardise(arr, mean, std):
    return (arr-mean)/std

def transform_back(arr, mean, std):
    return arr*std+mean

  
                 
def light_track_metric(targets, predictions, k =100):
    """
    RMSE based Metric for light track. Compare quartiles between MCMC-based methods and model output"
    targets: The reference quartiles generated from a MCMC technique (N x 3 x num_targets,)
    predictions: The quartiles predicted by  ( N x 3 x num_targets,)
    k: constant , used to adjust the magnitude of the score. Default = 100
    
    """
    targets = targets.flatten()
    predictions = predictions.flatten()
    scaled_x = targets/targets
    scaled_x_hat = predictions/targets
    score= k*(10-np.sqrt(((scaled_x - scaled_x_hat) ** 2).mean()))
    print("score is:",score)
    return score
                 
                 
def load_Quartile_Table(path, order= None):
    """Read quartiles information from Quartiles Table and generate a 3D matrix 
    Args:
        path (string): path to quartiles table
        order (list, optional): order of the parameters, there is a default order if order is not given Defaults to None.
    Returns:
        _type_: quartiles matrix used for calculating the light track metric (N, 3, num_targets)
    """
    import pandas as pd
    quartiles = pd.read_csv(path)
    if order is None:
        targets = ['T','log_H2O', 'log_CO2','log_CH4','log_CO','log_NH3']
    else:
        targets = order
    quartiles_matrix =  np.zeros((len(quartiles), 3, len(targets)))
    for t_idx, t in enumerate(targets):
        for q_idx, q in enumerate(['q1','q2','q3']):
            quartiles_matrix[:,q_idx, t_idx, ] = quartiles.loc[:,t + '_' + q]
    return quartiles_matrix





def to_light_track_format(q1_array, q2_array, q3_array, columns = None, name="LT_submission.csv"):
    """Helper function to prepare submission file for the light track, 
    we assume the test data is arranged in assending order of the planet ID.

    Args:
        q1_array: N x 6 array containing the estimates for 16% percentile
        q2_array: N x 6 array containing the estimates for 50% percentile
        q3_array: N x 6 array containing the estimates for 84% percentile
        columns: columns for the df. default to none

    Returns:
        Pandas DataFrame object
    """
    # create empty array
    LT_submission_df = pd.DataFrame(columns= columns)
    # sanity check - length should be equal
    assert len(q1_array) == len(q2_array) == len(q3_array)
    targets_label = ['T', 'log_H2O', 'log_CO2','log_CH4','log_CO','log_NH3']
    # create columns for df
    default_quartiles = ['q1','q2','q3']
    default_columns = []
    for c in targets_label:
        for q in default_quartiles:
            default_columns.append(c+q)
    
    if columns is None:
        columns = default_columns
    for i in tqdm(range(len(q1_array))):
        quartiles_dict = {}
        quartiles_dict['planet_ID'] = i
        for t_idx, t in enumerate(targets_label):
            quartiles_dict[f'{t}_q1']= q1_array[i, t_idx]
            quartiles_dict[f'{t}_q2']= q2_array[i, t_idx]
            quartiles_dict[f'{t}_q3']= q3_array[i, t_idx]
        LT_submission_df = pd.concat([LT_submission_df, pd.DataFrame.from_records([quartiles_dict])],axis=0,ignore_index = True)
    LT_submission_df.to_csv(name,index= False)
    return LT_submission_df


def to_regular_track_format(tracedata_arr, weights_arr, name="RT_submission.hdf5"):
    """convert input into regular track format.
    we assume the test data is arranged in assending order of the planet ID.

    Args:
        tracedata_arr (array): Tracedata array, usually in the form of N x M x 6, where M is the number of tracedata, here we assume tracedata is of equal size. It does not have to be but you will need to craete an alternative function if the size is different. 
        weights_arr (array): Weights array, usually in the form of N x M, here we assumed the number of weights is of equal size, it should have the same size as the tracedata

    Returns:
        None
    """
    submit_file = name
    RT_submission = h5py.File(submit_file,'w')
    for n in range(len(tracedata_arr)):
        ## sanity check - samples count should be the same for both
        assert len(tracedata_arr[n]) == len(weights_arr[n])
        ## sanity check - weights must be able to sum to one.
        assert np.isclose(np.sum(weights_arr[n]),1)

        grp = RT_submission.create_group(f"Planet_{n}")
        pl_id = grp.attrs['ID'] = n 
        tracedata = grp.create_dataset('tracedata',data=tracedata_arr[n])         
        weight_adjusted = weights_arr[n]

        weights = grp.create_dataset('weights',data=weight_adjusted)
    RT_submission.close()