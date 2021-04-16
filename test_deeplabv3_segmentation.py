import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2

from keras_deeplab_v3_plus.model import Deeplabv3
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input

from general_utils import GeneralUtils
import argparse

def main(weights, img_path):
    trained_image_width = 512
    image = GeneralUtils.get_image(img_path) #cv2.imread(r'C:\Users\m\Desktop\Image Segmentation\keras_deeplab_v3_plus\imgs\image1.jpg')

    # resize to max dimension of images from training dataset
    w, h, _ = image.shape
    ratio = float(trained_image_width) / np.max([w, h])

    temp_img = cv2.resize(image, (int(ratio * h), int(ratio * w)))

    resized_image = np.array(temp_img)
    resized_image.resize((512, 512, 3))

    resized_image = preprocess_input(resized_image, mode='tf')
    print("resized_image", resized_image.shape)

    deeplab_model = Deeplabv3()

    # coreml_model = coremltools.convert(deeplab_model,
    #                                    inputs=[coremltools.ImageType()],
    #                                    output_names=["output_layer"],
    #                                    image_input_names="input_1"
    # )
    #
    # start_time = time.time()
    #
    labels = deeplab_model.predict(np.expand_dims(resized_image, 0))

    # print("Prediction time:", time.time() - start_time)
    # print("res shape", res.shape)
    # print("res[0][0][0]", res[0][0][0].shape)
    # print("res[0][0][0]", res[0][0][0])
    #

    # squeezed_res = res.squeeze()
    # print("res.squeeze() shape", squeezed_res.shape)
    # labels = np.argmax(res, -1)
    # print(labels.shape)
    # remove padding and resize back to original image
    # if pad_x > 0:
    #     labels = labels[:-pad_x]
    # if pad_y > 0:
    #     labels = labels[:, :-pad_y]
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
    cv2.imshow("original img", image)
    cv2.waitKey(0)

    plt.imshow(labels)
    plt.waitforbuttonpress()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str)
    parser.add_argument('--img_path', type=str)

    args = parser.parse_args()

    main(args.weights, args.img_path)
