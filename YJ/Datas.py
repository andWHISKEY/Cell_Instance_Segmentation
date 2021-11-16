import os
import numpy as np
import json
import cv2
import rasterio
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pandas as pd
from utils import *

from torch.utils.data import Dataset

BASE_PATH = '/Users/youngjaepark/Desktop/PSL/Cell_Instance_Segmentation/sartorius-cell-instance-segmentation'
SHSY5Y_ANNOTATION_PATH = "/Users/youngjaepark/Desktop/PSL/Cell_Instance_Segmentation/sartorius-cell-instance-segmentation/LIVECell_dataset_2021/annotations/LIVECell_single_cells/shsy5y/livecell_shsy5y_train.json"
SHSY5Y_TRAIN_VAL_IMG_PATH = '/Users/youngjaepark/Desktop/PSL/Cell_Instance_Segmentation/sartorius-cell-instance-segmentation/LIVECell_dataset_2021/images/livecell_train_val_images'
IMAGE_RESIZE = (224, 224)

class CellDataset(Dataset):
    def __init__(self):
        # GET SATORIOUS
        sato_path = os.path.join(BASE_PATH, 'train')
        train_df = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
        self.ids = np.array(train_df.id.unique())

        # load sato images and masks
        id_list = []
        img_list = []
        mask_list = []
        for id in self.ids:
            id_list.append(id)
            img = cv2.imread(os.path.join(sato_path,str(id)+'.png'))
            mask = build_masks(train_df, id, input_shape=(520,704))
            mask = (mask>=1).astype('float32')

            # TO CHECK
            # plt.imshow(img, alpha=0.9)
            # plt.imshow(mask, alpha=0.1)
            # plt.show()

            img_list.append(img)
            mask_list.append(mask)
        # print(len(id_list), len(img_list), len(mask_list))

        # GET LIVE CELL DATAS
        with open(SHSY5Y_ANNOTATION_PATH) as f:
            data = json.load(f)
        ids = list()
        for i, img_dict in enumerate(data['images']):
            ids.append(data['images'][i]["id"])

        # Initialize dictionary
        d = {k: {'segmentation': [], 'bbox': [], 'path': []} for k in ids}
        for i in range(len(d)):
            d[data['images'][i]['id']]['path'].append(data['images'][i]['file_name'])

        for key in data['annotations'].keys():
            id = data['annotations'][key]['image_id']
            seg = data["annotations"][key]["segmentation"][0]
            bbox = data["annotations"][key]["bbox"]

            d[id]["segmentation"].append(seg)
            d[id]["bbox"].append(bbox)

        for id in ids:
            imew = os.path.join(SHSY5Y_TRAIN_VAL_IMG_PATH, 'SHSY5Y', d[id]['path'][0]) #[:-4]+'.tif')
            imew = rasterio.open(imew)
            imew = imew.read(1)

            array = np.zeros((520,704))
            for img_mask in d[id]['segmentation']:
                x = img_mask[0::2]
                y = img_mask[1::2]
                arr = [(x,y) for (x,y) in zip(y,x)]
                vertices = np.asarray(arr)
                path = Path(vertices)
                xmin, ymin, xmax, ymax = np.asarray(path.get_extents(), dtype=int).ravel()
                x, y = np.mgrid[:520, :704]

                points = np.vstack((x.ravel(), y.ravel())).T

                mask = path.contains_points(points)
                path_points = points[np.where(mask)]

                img_mask = mask.reshape(x.shape)
                img_mask = img_mask.astype(np.int)
                array += img_mask

            # plt.imshow(array)#, alpha=0.5)
            # #plt.imshow(imew, cmap='gray', alpha=0.9)
            # plt.show()

            id_list.append(id)
            img_list.append(imew)
            mask_list.append(array)
        print(len(id_list))


    def __getitem__(self, item):
        pass

    def __len__(self):
        pass



if __name__ == '__main__':
    train_dataset = CellDataset()
