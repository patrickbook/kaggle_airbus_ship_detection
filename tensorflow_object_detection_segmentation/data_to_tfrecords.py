from sklearn.utils import shuffle
import pandas as pd
import time
import tensorflow as tf
import os
import sys
import cv2
import io
import PIL.Image
import hashlib
from google.protobuf import text_format
import math
import numpy as np
import multiprocessing

sys.path.append("/home/research/DataServer/workspace/zhiyuan/projects/personal_projects/models/research/")
sys.path.append("/home/research/DataServer/workspace/zhiyuan/projects/personal_projects/models/research/object_detection")
from object_detection.protos import string_int_label_map_pb2

from utils import label_map_util


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_dataset_filename(dataset_dir, shard_id, NUM_SHARDS):
    output_filename = 'documents_%s_%05d-of-%05d.tfrecord' % (
        'train', shard_id, NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def bbox1(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    ymin = bbox[0]
    xmin = bbox[2]
    ymax = bbox[1]
    xmax = bbox[3]
    return [xmin,ymin,xmax,ymax]

def process_one_shard(shard_id, NUM_SHARDS, tf_record_data_path,num_per_shard,df_combined):
    with tf.Graph().as_default():
        image_reader = ImageReader()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            output_filename = _get_dataset_filename(
                tf_record_data_path, shard_id, NUM_SHARDS)

            length_files = len(df_combined)
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id + 1) * num_per_shard, length_files)
                for i in range(start_ndx, end_ndx):
                    sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                        i + 1, length_files, shard_id))
                    sys.stdout.flush()
                    try:
                        img_path = df_combined.iloc[i, :]['image_path']
                        with tf.gfile.GFile(img_path, 'rb') as fid:
                            encoded_jpg = fid.read()
                        encoded_jpg_io = io.BytesIO(encoded_jpg)
                        # image = PIL.Image.open(encoded_jpg_io)
                        # import pdb;
                        # pdb.set_trace()
                        height, width = image_reader.read_image_dims(sess, encoded_jpg)

                        key = hashlib.sha256(encoded_jpg).hexdigest()

                        xmins = []
                        ymins = []
                        xmaxs = []
                        ymaxs = []
                        classes = []
                        classes_text = []
                        # truncated = []
                        # poses = []
                        # difficult_obj = []
                        # masks = []
                        # ignore_difficult_instances = 0

                        # boxes = []
                        # scores = []
                        img_masks = df_combined.iloc[i,:]['masks']
                        tf_masks=[]
                        for mask in img_masks:
                            # all_masks += rle_decode(mask)

                            class_name = df_combined.iloc[i,:]['category']
                            classes_text.append(class_name.encode('utf8'))
                            classes.append(label_map_dict[class_name])
                            mask_mat=rle_decode(mask)
                            box = bbox1(mask_mat)
                            xmin = float(box[0])
                            ymin = float(box[1])
                            xmax = float(box[2])
                            ymax = float(box[3])

                            xmins.append(xmin / width)
                            ymins.append(ymin / height)
                            xmaxs.append(xmax / width)
                            ymaxs.append(ymax / height)
                            mask=mask_mat.astype(np.uint8)
                            tf_masks.append(mask)

                        feature_dict = {
                            'image/height': int64_feature(height),
                            'image/width': int64_feature(width),
                            'image/filename': bytes_feature(
                                img_path.encode('utf8')),
                            'image/source_id': bytes_feature(
                                img_path.encode('utf8')),
                            'image/key/sha256': bytes_feature(key.encode('utf8')),
                            'image/encoded': bytes_feature(encoded_jpg),
                            'image/format': bytes_feature('jpeg'.encode('utf8')),
                            'image/object/bbox/xmin': float_list_feature(xmins),
                            'image/object/bbox/xmax': float_list_feature(xmaxs),
                            'image/object/bbox/ymin': float_list_feature(ymins),
                            'image/object/bbox/ymax': float_list_feature(ymaxs),
                            'image/object/class/text': bytes_list_feature(classes_text),
                            'image/object/class/label': int64_list_feature(classes),
                            # 'image/object/difficult': int64_list_feature(difficult_obj),
                            # 'image/object/truncated': int64_list_feature(truncated),
                            # 'image/object/view': bytes_list_feature(poses),
                        }
                        encoded_mask_png_list = []
                        for mask in tf_masks:
                            img = PIL.Image.fromarray(mask)
                            output = io.BytesIO()
                            img.save(output, format='PNG')
                            encoded_mask_png_list.append(output.getvalue())
                        feature_dict['image/object/mask'] = (
                            bytes_list_feature(encoded_mask_png_list))

                        tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

                        tfrecord_writer.write(tf_example.SerializeToString())


                    except Exception:
                        print(img_path)
                        pass



# train_data_path = '/home/research/DataServer/datasets/external/airbus_ship_detection/new_data/train'
# train_files = os.listdir(train_data_path)
# print(len(train_files))
#
# masks = pd.read_csv('/home/research/DataServer/datasets/external/airbus_ship_detection/new_data/train_ship_segmentations_v2.csv')
#
# df = pd.DataFrame(columns=['image_name','image_path','masks','category'])
#
# count=0
# for image_index, each_image in enumerate(train_files):
#     # if image_index>5000:
#     #     break
#     print(str(image_index) + '/' + str(len(train_files)))
#     image_path = os.path.join(train_data_path, each_image)
#     img_masks = masks.loc[masks['ImageId'] == each_image, 'EncodedPixels'].tolist()
#     # import pdb;pdb.set_trace()
#     if pd.isnull(img_masks[0]):
#        continue
#     df.loc[count] = [each_image, image_path, img_masks,'ship']
#     count = count + 1
#
# df = shuffle(df, random_state=int(time.time()))
# df = df.reset_index(drop=True)

#
# df = pd.read_pickle(panda_path)
# df = shuffle(df, random_state=int(time.time()))
# df = df.reset_index(drop=True)
#
# df_select = df[['super_category']]
# df_group = df_select.groupby(['super_category']).size().sort_values(ascending=False)
#
# df_group_rows = df.groupby(['super_category'])
#
# df_back = df_group_rows.get_group('back')
# df_passport = df_group_rows.get_group('passport')
# df_paper_document = df_group_rows.get_group('paper_document')
# df_paper_like_document = df_group_rows.get_group('paper_like_document')
# df_card = df_group_rows.get_group('card')
#
# # import pdb; pdb.set_trace()
#
# df_card = df_card.iloc[0:50000,:]
# df_combined = pd.concat([df_back, df_passport,df_paper_document,df_paper_like_document,df_card], axis=0)
#
# df_combined = shuffle(df_combined, random_state=1)
# df_combined = df_combined.reset_index(drop=True)

label_map_proto = string_int_label_map_pb2.StringIntLabelMap()
all_classes=['ship']
num_classes=len(all_classes)

for index,each_class in enumerate(all_classes):
    print each_class
    item = label_map_proto.item.add()
    item.id = index+1
    item.name = each_class

# print label_map_proto


label_map_string = text_format.MessageToString(label_map_proto)
label_map_path='/home/research/DataServer/datasets/external/airbus_ship_detection/label_map/label_map.pbtxt'
with tf.gfile.GFile(label_map_path, 'wb') as fid:
        fid.write(label_map_string)

label_map_dict = label_map_util.get_label_map_dict(label_map_path)
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=len(label_map_dict), use_display_name=True)
category_index = label_map_util.create_category_index(categories)


tf_record_data_path='/home/research/DataServer/datasets/external/airbus_ship_detection/tfrecords'

NUM_SHARDS = 20
length_files = len(df)

num_per_shard = int(math.ceil(length_files / float(NUM_SHARDS)))
mp = multiprocessing.Pool(NUM_SHARDS)
results = []
for shard_id in range(NUM_SHARDS):

    print('processing sharid:'+str(shard_id))
    results.append(mp.apply_async(process_one_shard, args=(shard_id, NUM_SHARDS, tf_record_data_path, num_per_shard, df,)))

mp.close()
mp.join()
print([result.get() for result in results])

# process_one_shard(0, NUM_SHARDS, tf_record_data_path, num_per_shard, df)

# import pdb; pdb.set_trace()