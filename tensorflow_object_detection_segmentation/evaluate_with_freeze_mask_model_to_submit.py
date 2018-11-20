import os
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import time

from utils import label_map_util

from object_detection.utils import ops as utils_ops


def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# def rle_encode(img):
#     '''
#     img: numpy array, 1 - mask, 0 - background
#     Returns run length as string formated
#     '''
#     pixels = img.T.flatten()
#     pixels = np.concatenate([[0], pixels, [0]])
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
#     runs[1::2] -= runs[::2]
#     return ' '.join(str(x) for x in runs)

frozen_model_path='/home/research/DataServer/datasets/external/airbus_ship_detection/pb_models/mask_rcnn_inception_resnet_v2_atrous_fulldata/'


all_folders=os.listdir(frozen_model_path)
PATH_TO_LABELS = '/home/research/DataServer/datasets/external/airbus_ship_detection/label_map/label_map.pbtxt'
NUM_CLASSES = 1

######   Loading label map  ######
label_map_dict = label_map_util.get_label_map_dict(PATH_TO_LABELS)
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=len(label_map_dict), use_display_name=True)
category_index = label_map_util.create_category_index(categories)

all_classes=label_map_dict.keys()
num_classes=len(all_classes)


test_data_path = '/home/research/DataServer/datasets/external/airbus_ship_detection/new_data/test'

sample_submission_path = '/home/research/DataServer/datasets/external/airbus_ship_detection/new_data/sample_submission_v2.csv'

df = pd.read_csv(sample_submission_path)
# df = shuffle(df, random_state=0)
# df.reset_index(inplace=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

iteration_number=800000
for each_folder in all_folders:
    each_model=frozen_model_path+'/'+each_folder+'/'+'frozen_inference_graph.pb'
    print each_model
    ###### Load a (frozen) Tensorflow model into memory ######
    print each_folder
    if int(each_folder.split('.')[0]) != iteration_number:
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
        with tf.Session(graph=detection_graph, config = config) as sess:
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

            image_shape = tf.shape(image_tensor)
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image_shape[1], image_shape[2])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)

            detection_masks_reframed = tf.expand_dims(
                detection_masks_reframed, 0)

            # image_shape = tf.shape(image)
            # import pdb;
            # pdb.set_trace()
            # shuffle(all_classes)

            for index, row in df.iterrows():
                print(str(index)+' / '+str(len(df)))


                image_path = test_data_path + '/' + row['ImageId']

                image = Image.open(image_path)
                im_width, im_height = image.size
                image_np =  np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


                image_np_expanded = np.expand_dims(image_np, axis=0)
                # start = time.time()
                (boxes, scores, classes, num, masks) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, real_num_detection,detection_masks_reframed],
                    feed_dict={image_tensor: image_np_expanded})
                # end = time.time()
                # print 'time: ' + str(end - start)
                # print num
                # print category_index[classes[0][0]]
                # import pdb; pdb.set_trace()

                top1_confident_score=scores[0][0]
                # print top1_confident_score
                all_masks = np.zeros((im_height, im_width), dtype=np.uint8)
                threshold=0.9
                for score_index, each_score in enumerate(scores[0]):
                    # print each_score

                    if each_score <threshold and score_index==0:
                        all_masks = np.zeros((im_height, im_width), dtype=np.uint8)
                        break

                    elif each_score>=threshold:
                        # import pdb;pdb.set_trace()
                        pred_mask = masks[0][score_index]
                        all_masks += pred_mask
                        # print np.unique(all_masks)
                    elif each_score<threshold and score_index>0:
                        break

                all_masks[all_masks > 1] = 1
                encodedpixels = rle_encode(all_masks)

                df.loc[index,'EncodedPixels']=encodedpixels


            df.to_csv('/home/research/DataServer/datasets/external/airbus_ship_detection/submission_csv/submission9_th0'+ str(threshold).split('.')[1] +'_full_'+str(iteration_number) +'.csv', index=False)





