import numpy as np
import pandas as pd
from skimage.data import imread
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def bbox1(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    ymin = bbox[0]
    xmin = bbox[2]
    ymax = bbox[1]
    xmax = bbox[3]
    return [xmin,ymin,xmax,ymax]


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

def print_pandas_tables(df):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.options.display.max_colwidth = 200
    print(df.head())

# train_data_path = '/media/dataserver_gpu/home/research/DataServer/datasets/external/airbus_ship_detection/new_data/train'
test_data_path = '/home/research/DataServer/datasets/external/airbus_ship_detection/new_data/test'

# train_files = os.listdir(train_data_path)
# print(len(train_files))
test_files = os.listdir(test_data_path)
print(len(test_files))

masks = pd.read_csv('/home/research/DataServer/datasets/external/airbus_ship_detection/submission_csv/submissionSepNoOver_th095_full_1028000.csv')
print_pandas_tables(masks)

# fig, axarr = plt.subplots(1, 3, figsize=(15, 40))

for image_index, each_image in enumerate(test_files):
   print(image_index)
   image_path = os.path.join(test_data_path, each_image)
   img_masks = masks.loc[masks['ImageId'] == each_image, 'EncodedPixels'].tolist()
   # print(len(img_masks))
   # continue
   # import pdb;pdb.set_trace()
   if sum(masks['ImageId'] == each_image)==1:
       continue
   # if pd.isna(img_masks[0]):
   #     continue
   # Take the individual ship masks and create a single mask array for all ships
   # all_masks = np.zeros((768, 768))
   # boxes=[]
   for mask in img_masks:
       if pd.isnull(mask):
           print each_image
           import pdb;

           pdb.set_trace()
       # mask_mat = rle_decode(mask)
       # all_masks += mask_mat
       # boxes.append(bbox1(mask_mat))

   # img = imread(image_path)
   #
   # axarr[0].axis('off')
   # axarr[1].axis('off')
   # axarr[2].axis('off')
   # axarr[0].imshow(img)
   # axarr[1].imshow(all_masks)
   # axarr[2].imshow(img)
   # axarr[2].imshow(all_masks, alpha=0.4)
   #
   # for each_box in boxes:
   #
   #
   #     width = each_box[2] - each_box[0]
   #     height = each_box[3] - each_box[1]
   #
   #     rect = patches.Rectangle((each_box[0], each_box[1]), width, height, linewidth=2,
   #                              edgecolor='r',
   #                              facecolor='none')
   #
   #     axarr[0].add_patch(rect)
   # plt.tight_layout(h_pad=0.1, w_pad=0.1)
   # plt.waitforbuttonpress()
   # axarr[0].clear()
   # axarr[1].clear()
   # axarr[2].clear()