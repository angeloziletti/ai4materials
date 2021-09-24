from scipy.spatial.distance import minkowski
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from numpy import linalg as LA
from math import sqrt
import itertools

def tanimoto(spectrum_A, spectrum_B, power_value):
    spectrum_A = np.array(spectrum_A).flatten()
    spectrum_B = np.array(spectrum_B).flatten()

    norm_A = LA.norm(spectrum_A)
    norm_B = LA.norm(spectrum_B)

    dot_AB = np.dot(spectrum_A, spectrum_B)

    tanimoto_coefficient = dot_AB / ( norm_A * norm_A + norm_B * norm_B - dot_AB )

    return tanimoto_coefficient

def euclidean_distance(spectrum_A,spectrum_B,power_value):
    spectrum_A=np.array(spectrum_A).flatten()
    spectrum_B=np.array(spectrum_B).flatten()
    
    #normalize according to the maximum value
    #max_A=max(spectrum_A)
    #max_B=max(spectrum_B)
    #print max_A,max_B
    #spectrum_A=np.array([x/max_A for x in spectrum_A.flatten()]).flatten()
    #spectrum_B=np.array([x/max_B for x in spectrum_B.flatten()]).flatten()
    
    #Normalize according to modulus of spectrum vectors
    norm_A=LA.norm(spectrum_A)
    norm_B=LA.norm(spectrum_B)
    spectrum_A=np.array([x/norm_A for x in spectrum_A.flatten()]).flatten()
    spectrum_B=np.array([x/norm_B for x in spectrum_B.flatten()]).flatten()    
    
    eucl_metric=LA.norm(np.subtract(spectrum_A,spectrum_B))
    
    eucl_metric=np.power(eucl_metric,power_value)
    
    return 1.-eucl_metric


def cosine_sim_given_spectra(spectrum_A,spectrum_B,power_value=1.):
    
    spectrum_A=np.array(spectrum_A).flatten()
    spectrum_B=np.array(spectrum_B).flatten()
    
    dot_product=np.power(np.dot(spectrum_A,spectrum_B),power_value)
    
    norm_A=sqrt(np.power(np.dot(spectrum_A,spectrum_A),power_value))
    norm_B=sqrt(np.power(np.dot(spectrum_B,spectrum_B),power_value))
    """
    norm_A=LA.norm(spectrum_A)
    norm_B=LA.norm(spectrum_B)
    """
    kernel=dot_product/(norm_A*norm_B)
    return kernel

def plot_2D_matrix_with_axis_labels(matrix, x_axis_labels, y_axis_labels, title, savefig_name, save_fig_type,vmax,vmin,cmap=matplotlib.cm.get_cmap('Purples'), show_sim_values=True, fontsize=5, fig_size=(10,10)):
    
    matrix=np.array(matrix)
    
    if vmax==None:
        vmax=max(np.array(matrix).flatten())
    if vmin==None:
        vmin=min(np.array(matrix).flatten())
        
    x_pos=np.arange(len(x_axis_labels))
    y_pos=np.arange(len(y_axis_labels))
    
    #plt.tight_layout()
    fig, ax = plt.subplots(figsize=fig_size)
    
    plt.imshow(matrix,vmax=vmax,vmin=vmin,cmap=cmap)
    plt.colorbar()
    
    #ax.xaxis.set_ticks_position('bottom') # need to call this after matshow, otherwise error
    plt.xticks(x_pos,x_axis_labels, rotation=90, fontsize=fontsize)
    plt.yticks(y_pos,y_axis_labels, fontsize=fontsize)
    plt.title(title, y=1.1)
    
    #Add text    
    if show_sim_values:
        fmt = '.2f'
        thresh = max(matrix.flatten()) * 0.5
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, format(matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black", fontsize=fontsize)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(savefig_name+save_fig_type)
    plt.close()






def similarity_metric(spectrum_A,spectrum_B,metric,power_value,p=3):
    """
    ARGUMENTS: spectrum_A,spectrum_B: signals/spectra to be compared
               metric: similarity metric to be used (possible choices: 'cosine_sim','euclidean','cross_correlation' (TODO last one incomplete))
               power_value: power to which result of similarity metric is raised
               
    RETURNS:   value of similarity metric
    
    
    """
    

    spectrum_A=np.array(spectrum_A).flatten()
    spectrum_B=np.array(spectrum_B).flatten()

    if metric=='cosine_sim':
        
        return cosine_sim_given_spectra(spectrum_A,spectrum_B,power_value)
    
    elif metric=='euclidean':
        
        return euclidean_distance(spectrum_A,spectrum_B,power_value)
        
    elif metric=='minkowski':
        zero_vector=np.zeros(spectrum_A.size)
        
        
        norm_A=minkowski(u=spectrum_A,v=zero_vector,p=p,w=None)
        norm_B=minkowski(u=spectrum_B,v=zero_vector,p=p,w=None)
        
        spectrum_A_normed=spectrum_A/norm_A
        spectrum_B_normed=spectrum_B/norm_B
        
        minkowski_AB=minkowski(u=spectrum_A_normed,v=spectrum_B_normed,p=p,w=None)
        return 1.-minkowski_AB
        
    elif metric=='cross_correlation':
        #Not sure if this doesnt give the same as skp since automatic mode is 'valid' .... https://docs.scipy.org/doc/numpy/reference/generated/numpy.correlate.html
        corr_A_B=np.correlate(spectrum_A,spectrum_B)
        corr_A_A=np.correlate(spectrum_A,spectrum_A)
        corr_B_B=np.correlate(spectrum_B,spectrum_B)
    
        normalized_corr=corr_A_B/(corr_A_A*corr_B_B)
        
        return np.power(normalized_corr,power_value)

    elif metric == 'tanimoto':
        return tanimoto(spectrum_A, spectrum_B, power_value)
        
    else:
        
        raise NotImplementedError("The chosen metric is not implemented.")












def compute_and_plot_cross_similarity(descriptors_A,descriptors_B,sim_metric,power_value,p,x_axis_labels,y_axis_labels,
                                      title,savefig_name,save_fig_type='.png', 
                                      vmax=None, vmin=None,
                                      cmap=matplotlib.cm.get_cmap('Purples'), show_sim_values=True,
                                      fontsize=5, fig_size=(10,10)):
    
    sim_matrix=[]

    for descriptor_1 in descriptors_A:
        
        sim_matrix_row=[]
        
        for descriptor_2 in descriptors_B:
        
            #coherence_matrix_row.append(coherence_kernel(descriptor_1,descriptor_2))
            sim_matrix_row.append(similarity_metric(descriptor_1,descriptor_2,sim_metric,power_value,p))
    
        #coherence_matrix.append(coherence_matrix_row)
        sim_matrix.append(sim_matrix_row)
    
    
    plot_2D_matrix_with_axis_labels(sim_matrix, x_axis_labels, y_axis_labels, title, savefig_name, save_fig_type, vmax, vmin, cmap, show_sim_values, fontsize, fig_size)
        
        
    return sim_matrix





