                      
                                                                       

              
import yaml
from fvcore.common.config import CfgNode as CfgNodeOri

from . import custom_config
def load_yaml_with_base(text: str, allow_unsafe: bool = False):
\
\
\
\
\
\
\
\
\
\
       
    cfg = yaml.load(text, Loader=yaml.FullLoader)
    return cfg
class CfgNode(CfgNodeOri):
    def merge_from_str(self, text, allow_unsafe=False):
        loaded_cfg = load_yaml_with_base(text, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)
        self.merge_from_other_cfg(loaded_cfg)

                                                                               
                   
                                                                               
_C = CfgNode()


                                                                                
                    
                                                                                
_C.BN = CfgNode()

                   
_C.BN.USE_PRECISE_STATS = False

                                              
_C.BN.NUM_BATCHES_PRECISE = 200

                                        
_C.BN.WEIGHT_DECAY = 0.0

                                                                           
_C.BN.NORM_TYPE = "batchnorm"

                                                                      
                                                                         
_C.BN.NUM_SPLITS = 1

                                                                               
                               
_C.BN.NUM_SYNC_DEVICES = 1


                                                                                
                   
                                                                                
_C.TRAIN = CfgNode()

                                              
_C.TRAIN.ENABLE = True

          
_C.TRAIN.DATASET = "kinetics"

                        
_C.TRAIN.BATCH_SIZE = 64

_C.TRAIN.SPLIT = "train_subset2.pth"
                                                       
_C.TRAIN.EVAL_PERIOD = 1

                                                       
_C.TRAIN.CHECKPOINT_PERIOD = 1

                                                      
_C.TRAIN.CHECKPOINT_PERIOD_BY_ITER = 500


                                                                     
_C.TRAIN.AUTO_RESUME = True

                                                    
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

                                                 
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

                                                     
_C.TRAIN.CHECKPOINT_INFLATE = False


                                                                                
                 
                                                                                
_C.TEST = CfgNode()

                                                
_C.TEST.ENABLE = True

                      
_C.TEST.DATASET = "kinetics"

_C.TEST.SPLIT = "test_subset2.pth"
                       
_C.TEST.BATCH_SIZE = 8

                                                    
_C.TEST.CHECKPOINT_FILE_PATH = ""

                                                                      
                     
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

                                                                      
                     
_C.TEST.NUM_SPATIAL_CROPS = 3

                                                 
_C.TEST.CHECKPOINT_TYPE = "pytorch"
                                         
_C.TEST.SAVE_RESULTS_PATH = ""
                                                                               
                
                                                                               
_C.RESNET = CfgNode()

                          
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

                                                                 
_C.RESNET.NUM_GROUPS = 1

                                                   
_C.RESNET.WIDTH_PER_GROUP = 64

                                 
_C.RESNET.INPLACE_RELU = True

                           
_C.RESNET.STRIDE_1X1 = False

                                                                       
_C.RESNET.ZERO_INIT_FINAL_BN = False

                          
_C.RESNET.DEPTH = 50


                  
_C.RESNET.LABELS = ["continus","discontinus"]

                                                                               
                                         
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

                                         
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

                                           
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]


                                                                               
                  
                                                                               
_C.NONLOCAL = CfgNode()

                                                       
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

                                              
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

                                          
_C.NONLOCAL.INSTANTIATION = "dot_product"


                                           
_C.NONLOCAL.POOL = [
          
    [[1, 2, 2], [1, 2, 2]],
          
    [[1, 2, 2], [1, 2, 2]],
          
    [[1, 2, 2], [1, 2, 2]],
          
    [[1, 2, 2], [1, 2, 2]],
]

                                                                               
               
                                                                               
_C.MODEL = CfgNode()

                     
_C.MODEL.ARCH = "slowfast"

            
_C.MODEL.MODEL_NAME = "SlowFast"

                                                 
_C.MODEL.NUM_CLASSES = 400

                
_C.MODEL.LOSS_FUNC = "cross_entropy"

_C.MODEL.MASK_WEIGHT = 100

_C.MODEL.CLASS_WEIGHT = 1

                                                  
_C.MODEL.SINGLE_PATHWAY_ARCH = ["c2d", "i3d", "slow"]

                                                 
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

                                                       
_C.MODEL.DROPOUT_RATE = 0.5

                                        
_C.MODEL.FC_INIT_STD = 0.01

                                       
_C.MODEL.HEAD_ACT = "softmax"


                                                                               
                  
                                                                               
_C.SLOWFAST = CfgNode()

                                                                            
                             
_C.SLOWFAST.BETA_INV = 8

                                                                              
                
_C.SLOWFAST.ALPHA = 8

                                                                 
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

                                                                        
          
_C.SLOWFAST.FUSION_KERNEL_SZ = 5


                                                                               
              
                                                                               
_C.DATA = CfgNode()

                                 
_C.DATA.PATH_TO_DATA_DIR = ""

_C.DATA.DATASET = "faceforensics"

_C.DATA.MODE = ""

_C.DATA.ADAPTIVE = False

_C.DATA.SCALE = 1.0
                                            
_C.DATA.PATH_LABEL_SEPARATOR = " "

                           
_C.DATA.PATH_PREFIX = ""

                                          
_C.DATA.CROP_SIZE = 224

                                         
_C.DATA.NUM_FRAMES = 8

_C.DATA.NUM_FRAMES_RANGE = [1,2,3,4,5,6,7,8]

                                            
_C.DATA.SAMPLING_RATE = 8

                                                                   
_C.DATA.MEAN = [0.45, 0.45, 0.45]
                                         

_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

                                                                  
_C.DATA.STD = [0.225, 0.225, 0.225]

                                                      
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

                                     
_C.DATA.TRAIN_CROP_SIZE = 224

                                    
_C.DATA.TEST_CROP_SIZE = 256

                                                                               
                 
_C.DATA.TARGET_FPS = 30

                                                           
_C.DATA.DECODING_BACKEND = "pyav"

                                                                        
                                                                   
                         
_C.DATA.INV_UNIFORM_SAMPLE = False

                                                                              
_C.DATA.RANDOM_FLIP = True

                                        
_C.DATA.MULTI_LABEL = False

                                                                  
_C.DATA.ENSEMBLE_METHOD = "sum"

                                                          
_C.DATA.REVERSE_INPUT_CHANNEL = False


                                                                                
                   
                                                                                
_C.SOLVER = CfgNode()

                     
_C.SOLVER.BASE_LR = 0.1

                                                                         
_C.SOLVER.LR_POLICY = "cosine"

                           
_C.SOLVER.GAMMA = 0.1

                                                     
_C.SOLVER.STEP_SIZE = 1

                                          
_C.SOLVER.STEPS = []

                                       
_C.SOLVER.LRS = []

                           
_C.SOLVER.MAX_EPOCH = 300

           
_C.SOLVER.MOMENTUM = 0.9

                     
_C.SOLVER.DAMPENING = 0.0

                    
_C.SOLVER.NESTEROV = True

                    
_C.SOLVER.WEIGHT_DECAY = 1e-4

                                                               
_C.SOLVER.WARMUP_FACTOR = 0.1

                                                                  
_C.SOLVER.WARMUP_EPOCHS = 0.0

                                         
_C.SOLVER.WARMUP_START_LR = 0.01

                      
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

_C.SOLVER.LR_STEP = 50000

_C.SOLVER.TOTAL_STEP = 200000

_C.SOLVER.FREEZE_STEP = 10000


                                                                                
              
                                                                                

                                                               
_C.NUM_GPUS = 1

                                       
_C.NUM_SHARDS = 1

                                   
_C.SHARD_ID = 0

                 
_C.OUTPUT_DIR = "./tmp"

              
_C.TRAIN_MODULE= "train_unet_by_iter"

                                                                         
                                                     
_C.RNG_SEED = 1

                      
_C.LOG_PERIOD = 10

                              
_C.LOG_MODEL_INFO = True

                      
_C.DIST_BACKEND = "nccl"

                                                                                
                   
                                                                                
_C.BENCHMARK = CfgNode()

                                              
_C.BENCHMARK.NUM_EPOCHS = 5

                                                 
_C.BENCHMARK.LOG_PERIOD = 100

                                                         
_C.BENCHMARK.SHUFFLE = True


                                                                                
                                       
                                                                                
_C.DATA_LOADER = CfgNode()

                                                     
_C.DATA_LOADER.NUM_WORKERS = 8

                                  
_C.DATA_LOADER.PIN_MEMORY = True

                               
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False


                                                                                
                    
                                                                                
_C.DETECTION = CfgNode()

                                 
_C.DETECTION.ENABLE = False

                                                                                     
_C.DETECTION.ALIGNED = True

                       
_C.DETECTION.SPATIAL_SCALE_FACTOR = 16

                               
_C.DETECTION.ROI_XFORM_RESOLUTION = 7


                                                                               
                     
                                                                               
_C.AVA = CfgNode()

                           
_C.AVA.FRAME_DIR = "/mnt/fair-flash3-east/ava_trainval_frames.img/"

                                          
_C.AVA.FRAME_LIST_DIR = (
    "/mnt/vol/gfsai-flash3-east/ai-group/users/haoqifan/ava/frame_list/"
)

                                      
_C.AVA.ANNOTATION_DIR = (
    "/mnt/vol/gfsai-flash3-east/ai-group/users/haoqifan/ava/frame_list/"
)

                                           
_C.AVA.TRAIN_LISTS = ["train.csv"]

                                       
_C.AVA.TEST_LISTS = ["val.csv"]

                                                                           
                                                                      
           
_C.AVA.TRAIN_GT_BOX_LISTS = ["ava_train_v2.2.csv"]
_C.AVA.TRAIN_PREDICT_BOX_LISTS = []

                                       
_C.AVA.TEST_PREDICT_BOX_LISTS = ["ava_val_predicted_boxes.csv"]

                                                                          
_C.AVA.DETECTION_SCORE_THRESH = 0.9

                                           
_C.AVA.BGR = False

                                  
                                           
_C.AVA.TRAIN_USE_COLOR_AUGMENTATION = False

                                                                           
                                                      
_C.AVA.TRAIN_PCA_JITTER_ONLY = True

                                                       
_C.AVA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

                                 
_C.AVA.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

                                                
_C.AVA.TEST_FORCE_FLIP = False

                                                    
_C.AVA.FULL_TEST_ON_VAL = False

                                            
_C.AVA.LABEL_MAP_FILE = "ava_action_list_v2.2_for_activitynet_2019.pbtxt"

                                            
_C.AVA.EXCLUSION_FILE = "ava_val_excluded_timestamps_v2.2.csv"

                                              
_C.AVA.GROUNDTRUTH_FILE = "ava_val_v2.2.csv"

                                                         
_C.AVA.IMG_PROC_BACKEND = "cv2"

                                                                                
                            
                                                                            
                                                                                
_C.MULTIGRID = CfgNode()

                                                                              
                                                                    
                                                                         
_C.MULTIGRID.EPOCH_FACTOR = 1.5

                      
_C.MULTIGRID.SHORT_CYCLE = False
                                                                              
_C.MULTIGRID.SHORT_CYCLE_FACTORS = [0.5, 0.5 ** 0.5]

_C.MULTIGRID.LONG_CYCLE = False
                                                               
_C.MULTIGRID.LONG_CYCLE_FACTORS = [
    (0.25, 0.5 ** 0.5),
    (0.5, 0.5 ** 0.5),
    (0.5, 1),
    (1, 1),
]

                                                                  
                                                                           
                                                   
_C.MULTIGRID.BN_BASE_SIZE = 8

                                                                           
                                                                     
                                                                             
                                                                         
              
_C.MULTIGRID.EVAL_FREQ = 3

                                                                     
_C.MULTIGRID.LONG_CYCLE_SAMPLING_RATE = 0
_C.MULTIGRID.DEFAULT_B = 0
_C.MULTIGRID.DEFAULT_T = 0
_C.MULTIGRID.DEFAULT_S = 0

                                                                               
                                   
                                                                               
_C.TENSORBOARD = CfgNode()

                                                 
                                             
_C.TENSORBOARD.ENABLE = False
                                                       
                                                            
_C.TENSORBOARD.PREDICTIONS_PATH = ""
                                         
                                                        
_C.TENSORBOARD.LOG_DIR = ""
                                                       
                                                              
                                                                
                                   
_C.TENSORBOARD.CLASS_NAMES_PATH = ""

                                                       
                                                                            
_C.TENSORBOARD.CATEGORIES_PATH = ""

                                              
_C.TENSORBOARD.CONFUSION_MATRIX = CfgNode()
                             
_C.TENSORBOARD.CONFUSION_MATRIX.ENABLE = False
                                                
_C.TENSORBOARD.CONFUSION_MATRIX.FIGSIZE = [8, 8]
                                              
                                                            
_C.TENSORBOARD.CONFUSION_MATRIX.SUBSET_PATH = ""

                                     
_C.TENSORBOARD.HISTOGRAM = CfgNode()
                       
_C.TENSORBOARD.HISTOGRAM.ENABLE = False
                                                 
                                                      
_C.TENSORBOARD.HISTOGRAM.SUBSET_PATH = ""
                                                               
                    
_C.TENSORBOARD.HISTOGRAM.TOPK = 10
                                        
_C.TENSORBOARD.HISTOGRAM.FIGSIZE = [8, 8]

                                                           
                                     
_C.TENSORBOARD.MODEL_VIS = CfgNode()

                                     
_C.TENSORBOARD.MODEL_VIS.ENABLE = False

                                           
_C.TENSORBOARD.MODEL_VIS.MODEL_WEIGHTS = False

                                               
_C.TENSORBOARD.MODEL_VIS.ACTIVATIONS = False

                                          
_C.TENSORBOARD.MODEL_VIS.INPUT_VIDEO = False


                                                                         
                                                                  
                                                                         
                                                                                
                                                                       
                                                                                     
                                                                    
_C.TENSORBOARD.MODEL_VIS.LAYER_LIST = []
                                     
_C.TENSORBOARD.MODEL_VIS.TOPK_PREDS = 1
                                                      
_C.TENSORBOARD.MODEL_VIS.COLORMAP = "Pastel2"
                                                      
                                     
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM = CfgNode()
                                                        
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.ENABLE = True
                                                                       
                       
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST = []
                                                                   
                                            
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.USE_TRUE_LABEL = False
                                                      
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.COLORMAP = "viridis"

                                                              
                                     
_C.TENSORBOARD.WRONG_PRED_VIS = CfgNode()
_C.TENSORBOARD.WRONG_PRED_VIS.ENABLE = False
                                                  
_C.TENSORBOARD.WRONG_PRED_VIS.TAG = "Incorrectly classified videos."
                                                                        
                                   
_C.TENSORBOARD.WRONG_PRED_VIS.SUBSET_PATH = ""



               
_C.JITTER = CfgNode()

_C.JITTER.ENABLE = False

_C.JITTER.CONTINUS_METHODS=["blend_diff_person","blend_downsampled","blend_same_person"]
_C.JITTER.DISCONTINUS_METHODS=["light", "rotate", "skip"]

_C.JITTER.STRONG_INNER_CLIP_MASK_JITTER= False

                                                                                
              
                                                                                
_C.DEMO = CfgNode()

                         
_C.DEMO.ENABLE = False

                                                       
                                                              
_C.DEMO.LABEL_FILE_PATH = ""

                                                            
                          
                                 
_C.DEMO.WEBCAM = -1

                               
_C.DEMO.INPUT_VIDEO = ""
                                            
_C.DEMO.DISPLAY_WIDTH = 0
                                             
_C.DEMO.DISPLAY_HEIGHT = 0
                                                          
                                
_C.DEMO.DETECTRON2_CFG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                                                                
_C.DEMO.DETECTRON2_WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
                                                                
_C.DEMO.DETECTRON2_THRESH = 0.9
                                                           
                                                            
                                                        
                                                                            
_C.DEMO.BUFFER_SIZE = 0
                                                                           
                                                                             
_C.DEMO.OUTPUT_FILE = ""
                                                          
                                                
_C.DEMO.OUTPUT_FPS = -1
                                                       
_C.DEMO.INPUT_FORMAT = "BGR"
                                                                                                        
_C.DEMO.CLIP_VIS_SIZE = 10
                                              
_C.DEMO.NUM_VIS_INSTANCES = 2

                                      
_C.DEMO.PREDS_BOXES = ""
                                                     
_C.DEMO.THREAD_ENABLE = False
                                                                                     
                                                                                     
                                                                                      
                                         
_C.DEMO.NUM_CLIPS_SKIP = 0
                                                  
_C.DEMO.GT_BOXES = ""
                                                             
_C.DEMO.STARTING_SECOND = 900
                                                        
_C.DEMO.FPS = 30
                                                                             
                            
_C.DEMO.VIS_MODE = "thres"
                                   
_C.DEMO.COMMON_CLASS_THRES = 0.7
                                                     
                                                
_C.DEMO.UNCOMMON_CLASS_THRES = 0.3
                                                     
                              
_C.DEMO.COMMON_CLASS_NAMES = [
    "watch (a person)",
    "talk to (e.g., self, a person, a group)",
    "listen to (a person)",
    "touch (an object)",
    "carry/hold (an object)",
    "walk",
    "sit",
    "lie/sleep",
    "bend/bow (at the waist)",
]
                                                                        
                                                                      
_C.DEMO.SLOWMO = 1

                                        
custom_config.add_custom_config(_C)


def _assert_and_infer_cfg(cfg):
                    
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
                       
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

                      
    assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0
    assert cfg.TEST.NUM_SPATIAL_CROPS == 3

                        
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

                         
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
\
\
       
    return _assert_and_infer_cfg(_C.clone())
