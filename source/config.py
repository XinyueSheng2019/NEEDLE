############### CONFIG FILE FOR NEEDLE ############### 



# DATA AND MODEL PATHS
BAND = 'mixed'
IMAGE_PATH = '../../../data/image_sets_v3'
HOST_PATH = '../../../data/host_info_r5_ext'
MAG_PATH = '../../../data/mag_sets_v4'
OUTPUT_PATH = '../model_with_data/' + BAND + '_band/mixed_nor1_add_disc_t_ext_20240628/'
MODEL_NAME = '_model_nor1_neurons_64_128_128_ranking_updated_lasair'
LABEL_PATH = '../model_labels/label_dict_equal_test.json'
CUSTOM_TEST_PATH = OUTPUT_PATH + 'slsn_tde_easy_0.75.json'

# MODEL INPUTS
SEED = 456
QUALITY_MODEL_PATH = '../../quality_classifier/models/bogus_model_without_zscale'
NO_DIFF = True
ONLY_COMPLETE = True
ADD_HOST = True
ONLY_HOSTED_OBJ = False
META_ONLY = False
OBJECT_WITH_HOST_PATH = None #'../model_with_data/r_band/v12_v2_3c_20231126/hash_table.json'
META_NORMALIZE = 1
FEATURE_RANKING_PATH = OUTPUT_PATH + 'updated_xgb_binary_ranking.json'

# MODEL ARCHITECTURE
NEURONS = [[64,3],[128,3], [128,3]]
RES_CNN_GROUP = None
BATCH_SIZE = 128
EPOCH = 500
LEARNING_RATE = 5e-5

NOTE = 'Mixed band meta - updated feature ranking, with host, lasair version.'