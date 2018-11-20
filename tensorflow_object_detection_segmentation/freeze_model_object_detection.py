import sys
import subprocess
import os
import tensorflow as tf
import shutil

checkpoints_dir='/home/research/DataServer/datasets/external/airbus_ship_detection/tf_checkpoints/mask_rcnn_inception_resnet_v2_atrous_fulldata'
pipeline_config_path='/home/research/DataServer/datasets/external/airbus_ship_detection/model_config/mask_rcnn_inception_resnet_v2_atrous_coco.config'


export_file_path='/home/research/DataServer/workspace/zhiyuan/projects/personal_projects/models/research/object_detection/export_inference_graph.py'
export_folder_path='/home/research/DataServer/datasets/external/airbus_ship_detection/pb_models/mask_rcnn_inception_resnet_v2_atrous_fulldata'

all_checkpoints = tf.train.get_checkpoint_state(checkpoints_dir)
all_checkpoints_paths = all_checkpoints.all_model_checkpoint_paths
print len(all_checkpoints_paths)


count_checkpoints=0
for each_checkpoint_path in all_checkpoints_paths:
    # if int(each_checkpoint_path.split('-')[-1])<760000:
    #     continue
    output_path=export_folder_path + '/' + each_checkpoint_path.split('-')[-1] + '.pb'
    if os.path.exists(output_path):
        continue
    command_line = 'python ' + export_file_path + ' --input_type image_tensor ' + ' --pipeline_config_path ' + pipeline_config_path + \
                   ' --trained_checkpoint_prefix ' + each_checkpoint_path + ' --output_directory ' + output_path
    print command_line
    os.system(command_line)

    shutil.rmtree(output_path+'/saved_model')
    os.remove(output_path+'/checkpoint')
    os.remove(output_path + '/model.ckpt.data-00000-of-00001')
    os.remove(output_path + '/model.ckpt.index')
    os.remove(output_path + '/model.ckpt.meta')
    os.remove(output_path + '/pipeline.config')


