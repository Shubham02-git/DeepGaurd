import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def replace_name(csv_name: str):
    if csv_name == 'test_FaceForensics++_auc':
        return_name = 'FF++_c23'
    elif csv_name == 'test_FaceForensics++_c40_auc':
        return_name = 'FF++_c40'
    elif csv_name == 'test_Celeb-DF-v2_auc':
        return_name = 'CelebDF-v2'
    elif csv_name == 'test_DeepFakeDetection_auc':
        return_name = 'DFD'
    elif csv_name == 'test_DFDCP_auc':
        return_name = 'DFDCP'
    else:
        raise ValueError(f'Unknown csv name: {csv_name}')
    return return_name

detectors = glob.glob(os.path.join('aug_exp/*'))
results = []

for detector in detectors:
    for train_data in glob.glob(f'{detector}/*'):
        train_data_name = os.path.basename(train_data)
        csv_files = glob.glob(f'{train_data}/*.csv')
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            top_3_auc = df['Value'].nlargest(3).mean() 
            test_data = os.path.basename(csv_file).replace('.csv','') 
            model_name = os.path.basename(os.path.dirname(detector))
            results.append({
                'Model': model_name,
                'Detector': detector.replace('aug_exp/', ''),
                'Augmentation Methods': train_data_name,
                'Test Data': replace_name(test_data),
                'Top-3 AUC': top_3_auc
            })

import matplotlib as mpl

          
mpl.rcParams['font.size'] = 15                       

df_results = pd.DataFrame(results)
final_df = df_results.pivot_table(index=['Model', 'Detector', 'Augmentation Methods'], columns='Test Data', values='Top-3 AUC')


arr1 = np.tile(final_df.loc[(slice(None), slice(None), 'w_All'), :].values[0], (final_df.shape[0]//2, 1))
arr2 = np.tile(final_df.loc[(slice(None), slice(None), 'w_All'), :].values[1], (final_df.shape[0]//2, 1))
final_df = final_df.loc[(slice(None), slice(None)), :] - np.concatenate((arr1, arr2), 0)

final_df.reset_index(inplace=True)


                                                     
n_rows = 2
n_cols = 5

                                                               
fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 10))

                                                    
axes = axes.flatten()

                          
test_datasets = ['FF++_c23', 'FF++_c40', 'CelebDF-v2', 'DFD', 'DFDCP']

                                 
aug_methods = ['wo_All', 'wo_ColorJitter', 'wo_Compression', 'wo_GaussianBlur', 'wo_HorizontalFlip', 'wo_IsotropicResize', 'wo_Rotate']

                            
colors = ['lightcoral', 'lightsalmon', 'sandybrown', 'darkorange', 'gold', 'yellowgreen', 'lightgreen']

                               
for i, detector in enumerate(['SPSL', 'Xception']):
                                                   
    detector_df = final_df[final_df['Detector'] == detector]
    
                                       
    for j, test_data in enumerate(test_datasets):
                                                                        
        test_data_df = detector_df[test_data]
        
                                                                         
        test_data_df = test_data_df.sort_values(ascending=False, inplace=False)
        
                                                                                   
        num_methods = len(test_data_df)
        
                                        
        y_ticks = np.arange(num_methods)
        
                               
        bars = axes[i * n_cols + j].barh(y_ticks, test_data_df, color=colors[:num_methods])
        
                                
        for bar in bars:
            width = bar.get_width()
            axes[i * n_cols + j].annotate(f'{width:.3f}', xy=(width, bar.get_y() + bar.get_height() / 2),
                                            xytext=(3, -7), textcoords='offset points', ha='left', va='center')
        
                                    
        axes[i * n_cols + j].set_yticks(y_ticks)
        axes[i * n_cols + j].set_yticklabels(detector_df['Augmentation Methods'].reindex(test_data_df.index))
        
                           
        axes[i * n_cols + j].axvline(0, color='black', linestyle='--')
        
                              
        axes[i * n_cols + j].set_xlabel('Difference (%) from Baseline (w_All)')
        
                       
        axes[i * n_cols + j].set_title(f'Detector: {detector} | Test Data: {test_data}')
        axes[i * n_cols + j].title.set_fontsize(16)                         
    
                                     
fig.tight_layout()

               
plt.savefig(f"all_results_aug_bar_plot_2.png")
