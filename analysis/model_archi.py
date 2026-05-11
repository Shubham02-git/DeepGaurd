import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
import os

import matplotlib as mpl

          
                                                       

def get_top3_avg_auc(csv_path):
    df = pd.read_csv(csv_path)
    return df["Value"].nlargest(3).mean()

                                    
                                                                                  

                                                  
                         

                                          
                                             
                                                

                              
                                                       
                                                                               

                                                   
                              

                                                                                                 

                                                                          
                                                         
                                                                                                                   

                                                                  
                                                                                                                     
                                                                                                             
                                                          

                                                    
                                                                          

                                        
                                    
                                                                                                      
                                    
                                      
                                                                                                      
                                                                                     
                   
                                     
                                                                                                      

                                 
                                                                  
                                        
                                    
                                    
                                                         
                                                       
                                                                                               

                      
                       

                                                                                                                  
                                                                       
                        
                                                  

def plot_radar_chart(df, methods):
    fig, axs = plt.subplots(1, len(methods), figsize=(6 * len(methods), 6), subplot_kw=dict(polar=True))

    for ax, method in zip(axs, methods):
                                           
        df_method = df[df['Method'] == method]

                            
        num_vars = len(df_method['Dataset'].unique())
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

                                                 
        angles += angles[:1]

        ax.set_title(f'Average Top-3 AUC for {method.upper()}', fontsize=16, fontweight='bold')

                                                                        
        for arch in df_method['Architecture'].unique():
            df_arch = df_method[df_method['Architecture'] == arch].copy()                                        

                                                                
            dataset_order = ['DFDCP', 'Celeb-DF-v2', 'FaceForensics++', 'FaceForensics++_c40', 'DeepFakeDetection']
            df_arch['Dataset'] = pd.Categorical(df_arch['Dataset'], categories=dataset_order, ordered=True)
            df_arch.sort_values('Dataset', inplace=True)

            values = df_arch['Top-3 AUC'].tolist()
            values += values[:1]                                        

            if arch == 'EfficientNet':
                color = 'tab:blue'
                ax.plot(angles, values, label=arch, color=color, linewidth=1.2)                     
            elif arch == 'ResNet':
                color = 'tab:orange'
                ax.plot(angles, values, label=arch, color=color, linewidth=1.2)                     
                ax.fill(angles, values, color=color, alpha=0.05)                   
            else:
                color = 'tab:green'
                ax.plot(angles, values, label=arch, color=color, linewidth=1.2)                     

                               
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(df_arch['Dataset'].tolist())
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))                            

                    
        ax.grid(True)

    fig.suptitle('Average Top-3 AUC Across Different Methods and Architectures', size=15, color='black', y=1.05)
    plt.tight_layout()
    plt.savefig('radar_chart_model_archi_2.png')

                                 
base_dir = 'architecture_explore'
methods = ['core', 'spsl', 'ucf', 'facexray']
arch_mapping = {
    'core_eff': 'EfficientNet', 'core_res': 'ResNet', 'core_xcep': 'Xception',
    'spsl_eff': 'EfficientNet', 'spsl_res': 'ResNet', 'spsl_xcep': 'Xception',
    'ucf_eff': 'EfficientNet', 'ucf_res': 'ResNet', 'ucf_xcep': 'Xception',
    'facexray_eff': 'EfficientNet', 'facexray_res': 'ResNet', 'facexray_xcep': 'Xception'
}
datasets = ['DFDCP', 'Celeb-DF-v2', 'FaceForensics++', 'FaceForensics++_c40', 'DeepFakeDetection']

results = []

for method in methods:
    method_dir = os.path.join(base_dir, method)
    for arch_dir in [d for d in os.listdir(method_dir) if d != '.DS_Store']:
        folder_path = os.path.join(method_dir, arch_dir)
        arch = arch_mapping[arch_dir]
        for dataset in datasets:
            csv_file = f'test_{dataset}_auc.csv'
            if csv_file in os.listdir(folder_path):
                avg_auc = get_top3_avg_auc(os.path.join(folder_path, csv_file))
                results.append({
                    'Method': method,
                    'Architecture': arch,
                    'Dataset': dataset,
                    'Top-3 AUC': avg_auc
                })

df_results = pd.DataFrame(results)

                                    
plot_radar_chart(df_results, methods)
