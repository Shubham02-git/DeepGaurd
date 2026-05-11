                      
                                                                       

                                                           

import re


def get_name_convert_func():
\
\
\
\
\
       
    pairs = [
                                                                      
                                                                             
        [
            r"^nonlocal_conv([0-9]+)_([0-9]+)_(.*)",
            r"s\1.pathway0_nonlocal\2_\3",
        ],
                                 
        [r"^(.*)_nonlocal([0-9]+)_(theta)(.*)", r"\1_nonlocal\2.conv_\3\4"],
                         
        [r"^(.*)_nonlocal([0-9]+)_(g)(.*)", r"\1_nonlocal\2.conv_\3\4"],
                             
        [r"^(.*)_nonlocal([0-9]+)_(phi)(.*)", r"\1_nonlocal\2.conv_\3\4"],
                             
        [r"^(.*)_nonlocal([0-9]+)_(out)(.*)", r"\1_nonlocal\2.conv_\3\4"],
                                                                      
        [r"^(.*)_nonlocal([0-9]+)_(bn)_(.*)", r"\1_nonlocal\2.\3.\4"],
                                                                      
                                                                      
        [r"^t_pool1_subsample_bn_(.*)", r"s1_fuse.bn.\1"],
                                                   
        [r"^t_pool1_subsample_(.*)", r"s1_fuse.conv_f2s.\1"],
                                                                           
        [
            r"^t_res([0-9]+)_([0-9]+)_branch2c_bn_subsample_bn_(.*)",
            r"s\1_fuse.bn.\3",
        ],
                                                   
        [
            r"^t_res([0-9]+)_([0-9]+)_branch2c_bn_subsample_(.*)",
            r"s\1_fuse.conv_f2s.\3",
        ],
                                                                      
                                                                      
        [
            r"^res([0-9]+)_([0-9]+)_branch([0-9]+)([a-z])_(.*)",
            r"s\1.pathway0_res\2.branch\3.\4_\5",
        ],
                                                   
        [r"^res_conv1_bn_(.*)", r"s1.pathway0_stem.bn.\1"],
                                                        
        [r"^conv1_(.*)", r"s1.pathway0_stem.conv.\1"],
                                                                 
        [
            r"^res([0-9]+)_([0-9]+)_branch([0-9]+)_(.*)",
            r"s\1.pathway0_res\2.branch\3_\4",
        ],
                                                  
        [r"^res_conv1_(.*)", r"s1.pathway0_stem.conv.\1"],
                                                                      
                                                                      
        [
            r"^t_res([0-9]+)_([0-9]+)_branch([0-9]+)([a-z])_(.*)",
            r"s\1.pathway1_res\2.branch\3.\4_\5",
        ],
                                                   
        [r"^t_res_conv1_bn_(.*)", r"s1.pathway1_stem.bn.\1"],
                                                        
        [r"^t_conv1_(.*)", r"s1.pathway1_stem.conv.\1"],
                                                                 
        [
            r"^t_res([0-9]+)_([0-9]+)_branch([0-9]+)_(.*)",
            r"s\1.pathway1_res\2.branch\3_\4",
        ],
                                                  
        [r"^t_res_conv1_(.*)", r"s1.pathway1_stem.conv.\1"],
                                                                      
                                   
        [r"pred_(.*)", r"head.projection.\1"],
                              
        [r"(.*)bn.b\Z", r"\1bn.bias"],
                              
        [r"(.*)bn.s\Z", r"\1bn.weight"],
                                     
        [r"(.*)bn.rm\Z", r"\1bn.running_mean"],
                                     
        [r"(.*)bn.riv\Z", r"\1bn.running_var"],
                         
        [r"(.*)[\._]b\Z", r"\1.bias"],
                           
        [r"(.*)[\._]w\Z", r"\1.weight"],
    ]

    def convert_caffe2_name_to_pytorch(caffe2_layer_name):
\
\
\
\
\
\
\
           
        for source, dest in pairs:
            caffe2_layer_name = re.sub(source, dest, caffe2_layer_name)
        return caffe2_layer_name

    return convert_caffe2_name_to_pytorch
