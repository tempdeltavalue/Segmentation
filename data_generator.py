import numpy as np
import os
import keras
import cv2
import json
import urllib.request

# from dataset_tool import DatasetTool

import time

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,
                 base_path,
                 batch_size,
                 is_val=False):

        ann_path = r"C:\Users\m\Desktop\COCOtestset\annotations\instances_val2017.json"

        with open(ann_path) as f:
            data = json.load(f)
            self.images = data["images"]
            self.annotations = data["annotations"]

        self.batch_size = batch_size

        # self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        batch_items = self.images[index * self.batch_size:(index + 1) * self.batch_size]

        X = np.empty((self.batch_size, 224, 224, 3))

        ind = 0
        while ind < self.batch_size:
            # item_path, class_id, img_id_key = batch_items_paths[ind]

            url = batch_items[ind]["coco_url"]

            try:
                with urllib.request.urlopen(url) as url:
                    s = url.read()

                    arr = np.asarray(bytearray(s), dtype=np.uint8)
                    img = cv2.imdecode(arr, -1)  # 'Load it as it is'
                    img = cv2.resize(img, (224, 224))
                    X[ind] = img

                ind += 1
            except Exception as e:
                print("Error", e)

        return X#, class_Y

    # def on_epoch_end(self):
    #     np.random.shuffle(self.items_paths)

def create_dg():
    is_val = False
    data_generator = DataGenerator(base_path="",batch_size=16,  is_val=is_val)
    test_batch = data_generator[0]

    return data_generator

def test_show_image():
    data_generator = create_dg()

    start_time = time.time()
    items = data_generator[0]
    print(items[1].shape)
    print(items[1])
    print("Item generation time:", time.time() - start_time)

    cv2.imshow("batch_image", items[0][1])
    cv2.waitKey(0)

if __name__ == '__main__':
    test_show_image()
    # test_class_dict()