import sys
import subprocess
import os


train_file_path='/home/research/DataServer/workspace/zhiyuan/projects/personal_projects/models/research/object_detection/model_main.py'
pipeline_config_path='/home/research/DataServer/datasets/external/airbus_ship_detection/model_config/mask_rcnn_inception_resnet_v2_atrous_coco.config'
model_dir='/home/research/DataServer/datasets/external/airbus_ship_detection/tf_checkpoints/mask_rcnn_inception_resnet_v2_atrous_fulldata'
NUM_TRAIN_STEPS= 5000000
num_eval_steps = 1
command_line='python '+ train_file_path + ' --pipeline_config_path ' + pipeline_config_path + \
            ' --model_dir ' + model_dir + ' --num_train_steps ' + str(NUM_TRAIN_STEPS) + \
             ' --num_eval_steps ' + str(num_eval_steps) + ' --alsologtostderr'


print command_line
# while True:
#     try:
os.system(command_line)
    # except Exception:
    #     print 'error'