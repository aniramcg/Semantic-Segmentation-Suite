import os
import cv2
import tensorflow as tf
import numpy as np
import time
import argparse
import random
import os
import SimpleITK as sitk
# use 'Agg' on matplotlib so that plots could be generated even without Xserver
# running
import matplotlib

from utils import utils, helpers
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

dataset = '/home/esgomezm/Documents/3D-PROTUCEL/data/SemmanticSeg'

if not os.path.exists('/home/esgomezm/Documents/3D-PROTUCEL/Semantic-Segmentation-Suite/data_visualization/'):
    os.makedirs('/home/esgomezm/Documents/3D-PROTUCEL/Semantic-Segmentation-Suite/data_visualization/')
batch_size = 3
crop_height = 256
crop_width = 256
h_flip = True
v_flip = True
brightness = None
rotation = 180

def data_augmentation(input_image, output_image):
    # Data augmentation
    # crop a patch sampling cell bodies with a sampling pdf (cell==1 has weight 10000 and cell == 0 has weight 1)


    if h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if brightness:
        factor = 1.0 + random.uniform(-1.0*args.brightness, args.brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if rotation:
        angle = random.uniform(-1*rotation, rotation)
    if rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)
    input_image, output_image = utils.random_crop(input_image, output_image, crop_height, crop_width)
    return input_image, output_image


# Get the names of the classes so we can record the evaluation results
class_names_list, label_values = helpers.get_label_info(os.path.join(dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=dataset)
for epoch in range(0, 1):

    current_losses = []

    cnt = 0

    # Equivalent to shuffling
    id_list = np.random.permutation(len(train_input_names))

    num_iters = int(np.floor(len(id_list) / batch_size))
    st = time.time()
    epoch_st = time.time()
    for i in range(num_iters):
        # st=time.time()

        input_image_batch = []
        output_image_batch = []

        # Collect a batch of images
        for j in range(batch_size):
            index = i * batch_size + j
            id = id_list[index]
            input_image = utils.load_image(train_input_names[id])

            output_image = utils.load_image(train_output_names[id])

            # Make output binary as our masks are instance masks
            if np.max(output_image) > np.max(label_values):
                output_image[output_image > 0] = 1

            with tf.device('/cpu:0'):
                input_image, output_image = data_augmentation(input_image, output_image)

                if len(input_image.shape) < 3:
                    input_image = input_image.reshape((input_image.shape[0], input_image.shape[1], 1))

                # Prep the data. Make sure the labels are in one-hot format
                # Our images are uint16
                # input_image = np.float32(input_image) / 255.0
                input_image = np.float32(input_image) / (2 ** (16) - 1)



                output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))

                input_image_batch.append(np.expand_dims(input_image, axis=0))
                output_image_batch.append(np.expand_dims(output_image, axis=0))

        if batch_size == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1), axis=0)
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1), axis=0)

        print('Input size {}'.format(input_image_batch.shape))
        print('Output size {}'.format(output_image_batch.shape))
        aux = 255*input_image_batch[:,:,:,0]
        aux = aux.astype(np.uint8)
        sitk.WriteImage(sitk.GetImageFromArray(aux),
                        '/home/esgomezm/Documents/3D-PROTUCEL/Semantic-Segmentation-Suite/data_visualization/input_{}.tif'.format(i))
        aux = 255 * output_image_batch[:, :, :, 1]
        aux = aux.astype(np.uint8)
        sitk.WriteImage(sitk.GetImageFromArray(aux),
                        '/home/esgomezm/Documents/3D-PROTUCEL/Semantic-Segmentation-Suite/data_visualization/output_{}.tif'.format(i))