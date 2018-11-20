import matplotlib
# matplotlib.use('Agg')
matplotlib.use('TkAgg')
import os
import tensorflow as tf
import sys
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.patches as patches
from sklearn.utils import shuffle
import cv2
import time
# sys.path.append("/media/dataserver/workspace/zhiyuan/projects/personal_projects/models/research/object_detection")
# sys.path.append("/media/dataserver/workspace/zhiyuan/projects/personal_projects/models/research")
#####  Object detection imports  #####

from utils import label_map_util
# tf.logging.set_verbosity(tf.logging.INFO)
# tf.logging.set_verbosity(tf.logging.ERROR)
from utils import visualization_utils as vis_util


# import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.WARN)

from object_detection.utils import ops as utils_ops

# export TF_CPP_MIN_LOG_LEVEL=3

# frozen_model_path='/media/dataserver/datasets/internal/zhiyuan/data/document_detection/models/model/export_pb'
frozen_model_path='/home/research/DataServer/datasets/external/airbus_ship_detection/pb_models/mask_rcnn_inception_resnet_v2_atrous'


all_folders=os.listdir(frozen_model_path)
PATH_TO_LABELS = '/home/research/DataServer/datasets/external/airbus_ship_detection/label_map/label_map.pbtxt'
NUM_CLASSES = 1

######   Loading label map  ######
label_map_dict = label_map_util.get_label_map_dict(PATH_TO_LABELS)
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=len(label_map_dict), use_display_name=True)
category_index = label_map_util.create_category_index(categories)

all_classes=label_map_dict.keys()
# test_image_path = '/media/dataserver/datasets/internal/zhiyuan/data/document_detection/test_data'
# # test_image_path='/media/dataserver/datasets/internal/Christos/Shared/For_Zhiyuan'
# all_classes=os.listdir(test_image_path)
num_classes=len(all_classes)


test_data_path = '/home/research/DataServer/datasets/external/airbus_ship_detection/new_data/test'

sample_submission_path = '/home/research/DataServer/datasets/external/airbus_ship_detection/new_data/sample_submission_v2.csv'

df = pd.read_csv(sample_submission_path)
# df = shuffle(df, random_state=0)
# df.reset_index(inplace=True)



for each_folder in all_folders:
    each_model=frozen_model_path+'/'+each_folder+'/'+'frozen_inference_graph.pb'
    print each_model
    ###### Load a (frozen) Tensorflow model into memory ######
    print each_folder
    # if int(each_folder.split('.')[0])!=295722:
    if int(each_folder.split('.')[0]) != 2000:
        continue

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(each_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            # for op in detection_graph.get_operations():
            #     print(op.name)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = tf.squeeze(detection_graph.get_tensor_by_name('detection_boxes:0'),[0])

            detection_masks=tf.squeeze(detection_graph.get_tensor_by_name('detection_masks:0'),[0])
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            real_num_detection = tf.cast(detection_graph.get_tensor_by_name('num_detections:0')[0],tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            # detection_masks = tf.expand_dims(detection_masks, 0)

            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

            # image_shape = tf.shape(image)
            # import pdb;
            #
            # pdb.set_trace()
            # shuffle(all_classes)

            for index, row in df.iterrows():



                image_path = test_data_path + '/' + row['ImageId']

                image = Image.open(image_path)
                im_width, im_height = image.size
                image_np=np.asarray(image)
                if image_np.ndim == 2:
                    img_b = image_np
                    image_np = np.zeros((im_height, im_width, 3))
                    image_np[:, :, 0] = img_b
                    image_np[:, :, 1] = img_b
                    image_np[:, :, 2] = img_b
                if image_np.shape[-1] == 4:
                    image_np = np.asarray(image.convert('RGB'))

                # print type(image_np)
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks[0:1,:], detection_boxes[0:1,:], image_np.shape[0], image_np.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)

                detection_masks_reframed = tf.expand_dims(
                    detection_masks_reframed, 0)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                start = time.time()
                (boxes, scores, classes, num, masks) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, real_num_detection,detection_masks_reframed],
                    feed_dict={image_tensor: image_np_expanded})
                end = time.time()
                print 'time: ' + str(end - start)
                print num
                # print category_index[classes[0][0]]
                # import pdb; pdb.set_trace()

                top1_confident_score=scores[0][0]
                print top1_confident_score
                if top1_confident_score <0.05:
                    continue

                fig, imgplot = plt.subplots(1)
                top1_class=category_index[classes[0][0]]['name']
                top1_bbox=boxes[0]
                # print scores
                # print scores[0][0]
                # print category_index[classes[0][1]]
                # print scores[0][1]
                # print boxes
                # import pdb;
                #
                # pdb.set_trace()
                pred_ymin=  top1_bbox[0]*im_height
                pred_xmin = top1_bbox[1]*im_width
                pred_ymax = top1_bbox[2]*im_height
                pred_xmax = top1_bbox[3]*im_width
                pred_width=pred_xmax-pred_xmin
                pred_height=pred_ymax-pred_ymin


                img_read = mpimg.imread(image_path)


                imgplot.imshow(img_read)
                # imgplot.imshow(masks[0][0])

                rect_pred = patches.Rectangle((pred_xmin, pred_ymin), pred_width, pred_height, linewidth=2,
                                         edgecolor='b',
                                         facecolor='none')

                imgplot.add_patch(rect_pred)
                plt.axis('off')
                # plt.title(top1_class)
                plt.imshow(masks[0][0], alpha=0.5)

                # plt.waitforbuttonpress()
                plt.show()
                # plt.cla()
                # count_file = count_file + 1


                # plt.show()
                # plt.waitforbuttonpress()




