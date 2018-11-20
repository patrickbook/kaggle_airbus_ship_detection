import os
import pandas as pd
from sklearn.utils import shuffle
import time

train_data_path = '/home/research/DataServer/datasets/external/airbus_ship_detection/new_data/train'
train_files = os.listdir(train_data_path)
print(len(train_files))

masks = pd.read_csv('/home/research/DataServer/datasets/external/airbus_ship_detection/new_data/train_ship_segmentations_v2.csv')

df = pd.DataFrame(columns=['image_name','image_path','masks','category'])

count=0
for image_index, each_image in enumerate(train_files):
    # if image_index>5000:
    #     break
    print(str(image_index) + '/' + str(len(train_files)))
    image_path = os.path.join(train_data_path, each_image)
    img_masks = masks.loc[masks['ImageId'] == each_image, 'EncodedPixels'].tolist()
    # import pdb; pdb.set_trace()
    if pd.isnull(img_masks[0]) and len(img_masks)==1:
        category='noship'
    elif pd.isnull(img_masks[0]) and len(img_masks)!=1:
        print 'errorrrrrrrrrrrrrrrrrrrrr'
        print img_masks
        break
    elif not pd.isnull(img_masks[0]):
        category='ship'

    df.loc[count] = [each_image, image_path, img_masks,category]
    count = count + 1

df = shuffle(df, random_state=int(time.time()))
df = df.reset_index(drop=True)
df_save_file_name = '/home/research/DataServer/datasets/external/airbus_ship_detection/panda_files/panda_training_data.pkl'
df.to_pickle(df_save_file_name)

