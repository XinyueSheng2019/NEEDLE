############### CONFIG FILE FOR NEEDLE ############### 



# DATA AND MODEL PATHS
BAND = 'r'
IMAGE_PATH = '../data/image_sets_v3'
HOST_PATH = '../data/host_info_r5'
MAG_PATH = '../data/mag_sets_v4'
OUTPUT_PATH = '../model_with_data/' + BAND + '_band/test_main/'
MODEL_NAME = '_model_'
LABEL_PATH = '../model_labels/label_dict_equal_test.json'
CUSTOM_TEST_PATH = None

# MODEL INPUTS
SEED = 456
QUALITY_MODEL_PATH = 'bogus_model_without_zscale'
NO_DIFF = True
ONLY_COMPLETE = True
ADD_HOST = True
ONLY_HOSTED_OBJ = False
META_ONLY = False
OBJECT_WITH_HOST_PATH = None #'../model_with_data/r_band/v12_v2_3c_20231126/hash_table.json'

# MODEL ARCHITECTURE
NEURONS = [[64,3],[128,3]]
RES_CNN_GROUP = None
BATCH_SIZE = 128
EPOCH = 300
LEARNING_RATE = 5e-5

