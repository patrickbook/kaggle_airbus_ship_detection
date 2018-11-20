import pandas as pd
from sklearn.utils import shuffle
import time
from skimage.data import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.draw import polygon
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.max_colwidth = 200


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    # import pdb;
    # pdb.set_trace()
    return colormap[label]


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap



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

def masks_as_independent(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros((768, 768), dtype = np.int8)
    # scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:,:] += (i+1) * rle_decode(mask)
            # import pdb;pdb.set_trace()
    return all_masks

def masks_as_all(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros((768, 768), dtype = np.int8)
    # scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:,:] |= rle_decode(mask)
            # import pdb;pdb.set_trace()
    return all_masks

pd_path = '/home/research/DataServer/datasets/external/airbus_ship_detection/submission_csv/submissionSepNoOverRectOneWithin_th095_full_800000.csv'
test_folder = '/home/research/DataServer/datasets/external/airbus_ship_detection/new_data/test'
df = pd.read_csv(pd_path)
# import pdb; pdb.set_trace()
print(df.head())
print(len(df))
# df_group = df.groupby(['category'])
# print df_group.size()
# df_noship = df_group.get_group('noship')
# df_ship = df_group.get_group('ship')


df = shuffle(df, random_state=int(time.time()))
df = df.reset_index(drop=True)

count = 0
for index, row in df.iterrows():
    print index
    if count>5:
        break
    image_path = test_folder + '/' + row['ImageId']
    encodedpixels = row['EncodedPixels']


    if pd.isnull(encodedpixels):
        continue

    fig, axarr = plt.subplots(1, 3, figsize=(10, 3))

    color_mask = np.zeros((768, 768), dtype=np.int8)
    if not pd.isnull(encodedpixels):
        mask = rle_decode(encodedpixels)
        color_mask = label_to_color_image(mask).astype(np.uint8)

        # import pdb;pdb.set_trace()

        _, contours, _ = cv2.findContours(mask.copy(), 1, 1)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        img_height = mask.shape[0]
        img_width = mask.shape[1]
        mask_rect = np.zeros((768, 768), dtype=np.uint8)
        newxs = [box[0][0],box[1][0],box[2][0],box[3][0]]
        newys = [box[0][1], box[1][1], box[2][1], box[3][1]]
        rr, cc = polygon(newys, newxs)

        rr_index = rr < img_height
        rr = rr[rr_index]
        cc = cc[rr_index]
        cc_index = cc < img_width
        cc = cc[cc_index]
        rr = rr[cc_index]

        mask_rect[rr, cc] = 1

    # import pdb;
    #
    # pdb.set_trace()
    img=imread(image_path)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    # axarr[2].axis('off')
    # axarr[0].imshow(img)
    # axarr[1].imshow(color_mask)
    # axarr[0].imshow(img)
    # axarr[0].imshow(color_mask, alpha=0.8)
    # axarr[1].imshow(img)
    axarr[0].imshow(img)
    axarr[1].imshow(color_mask)
    axarr[2].imshow(mask_rect)

    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    plt.show()
    count = count + 1


# df_group = df_select.groupby(['super_category']).size().sort_values(ascending=False)

# randomly selected ship and noship
# df_ship = shuffle(df_ship, random_state=int(time.time()))
# df_ship = df_ship.reset_index(drop=True)
#
#
# count=0
# for index, row in df_ship.iterrows():
#     print index
#     # import pdb; pdb.set_trace()
#     if count>5:
#         break
#
#
#
#     image_path = row['image_path']
#     if len(row['masks'])<=1:
#         continue
#     fig, axarr = plt.subplots(1, 3, figsize=(25, 7))
#     mask = masks_as_all(row['masks'])
#     color_mask = label_to_color_image(mask).astype(np.uint8)
#     # import pdb;
#     #
#     # pdb.set_trace()
#     img=imread(image_path)
#     axarr[0].axis('off')
#     axarr[1].axis('off')
#     axarr[2].axis('off')
#     axarr[0].imshow(img)
#     axarr[1].imshow(color_mask)
#     axarr[2].imshow(img)
#     axarr[2].imshow(color_mask, alpha=0.4)
#
#     plt.tight_layout(h_pad=0.1, w_pad=0.1)
#     plt.show()
#     count = count + 1



#
# df_noship = shuffle(df_noship, random_state=int(time.time()))
# df_noship = df_noship.reset_index(drop=True)
#
# count=0
# for index, row in df_noship.iterrows():
#     # print index
#     print row
#     # import pdb; pdb.set_trace()
#     if count>5:
#         break
#
#     image_path = row['image_path']
#
#     fig, axarr = plt.subplots(1, 1, figsize=(7, 7))
#
#     img=imread(image_path)
#     axarr.axis('off')
#     axarr.imshow(img)
#
#     plt.tight_layout(h_pad=0.1, w_pad=0.1)
#     plt.show()
#     count = count + 1