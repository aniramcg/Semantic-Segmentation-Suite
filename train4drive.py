from __future__ import print_function
import os, time, cv2, sys, math
import tensorflow as tf
import re
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess
from pandas import DataFrame

# use 'Agg' on matplotlib so that plots could be generated even without Xserver
# running
import matplotlib

matplotlib.use('Agg')

from utils import utils, helpers
from builders import model_builder

import matplotlib.pyplot as plt


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Create a DataFrame for saving later the values to an .xlsx file
# df = DataFrame(index=range(1), columns=['Average', 'Cell', 'Background', 'Validation precision','Validation recall','F1 score', 'IoU score', 'LR'])

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=20, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=100, help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=None,
                    help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=True,
                    help='Whether to continue training from a checkpoint. It will read the last epoch and keep working from that with epoch_start_i=last_epoch.')
parser.add_argument('--checkpoint', type=str, default=time.ctime(),
                    help='Name of the checkpoint to look for pretrained weights')
# parser.add_argument('--dataset', type=str, default="D:\MarinaCalzada\3dprotucell\RGB_data", help='Dataset you are using.')
# parser.add_argument('--dataset', type=str, default="/content/gdrive/My Drive/TFG/TFG MARINA CALZADA/clean_data/RGB_folder/", help='Dataset you are using.') #NEED TO PUT A FULL DIRECTORY, IT WON'T WORK ONLY PUTTING THE FOLDER
parser.add_argument('--dataset', type=str, default="/data/data/SemmanticSeg/",
                    help='Dataset you are using.')  # NEED TO PUT A FULL DIRECTORY, IT WON'T WORK ONLY PUTTING THE FOLDER
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--reduce_lr', type=str2bool, default=True,
                    help='Whether to reduce the learning rate during training')
parser.add_argument('--num_val_images', type=int, default=350, help='The number of images to used for validations')
parser.add_argument('--stor_val', type=str2bool, default=True,
                    help='Wether to store shots of the validation images used during the training')
parser.add_argument('--random_val', type=str2bool, default=False,
                    help='Wether to analyze a random set of validation images')
parser.add_argument('--h_flip', type=str2bool, default=True,
                    help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=True,
                    help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None,
                    help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--rotation', type=float, default=180,
                    help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
parser.add_argument('--model', type=str, default="MobileUNet",
                    help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="MobileNetV2",
                    help='The frontend you are using. See frontend_builder.py for supported models')
args = parser.parse_args()


def data_augmentation(input_image, output_image):
    # Data augmentation
    if args.h_flip and random.randint(0, 1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args.v_flip and random.randint(0, 1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args.brightness:
        factor = 1.0 + random.uniform(-1.0 * args.brightness, args.brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(-1 * args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1] // 2, input_image.shape[0] // 2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]),
                                     flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]),
                                      flags=cv2.INTER_NEAREST)

    # crop a patch sampling cell bodies with a sampling pdf (cell==1 has weight 10000 and cell == 0 has weight 1)
    input_image, output_image = utils.random_crop(input_image, output_image, args.crop_height, args.crop_width)
    return input_image, output_image


# Get the names of the classes so we can record the evaluation results
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Compute your softmax cross entropy loss
# net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_input = tf.placeholder(tf.float32, shape=[None, None, None, 1])  # We work with intensities not RGB images.
net_output = tf.placeholder(tf.float32, shape=[None, None, None, num_classes])

network, init_fn = model_builder.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input,
                                             num_classes=num_classes, crop_width=args.crop_width,
                                             crop_height=args.crop_height, is_training=True)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))
if args.reduce_lr == False:
    # opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate=args.learning_rate, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate, name='Adam').minimize(loss,
                                                                                                   var_list=[var for var
                                                                                                             in
                                                                                                             tf.trainable_variables()])
else:
    # Definition of the decay (decay_steps = 100000) : decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    starter_learning_rate = args.learning_rate
    learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96,
                                                         staircase=True)
    # Passing global_step to minimize() will increment it at each step.
    # learning_step = (tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(...my loss..., global_step=global_step))
    # opt_0 = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()], global_step = global_step)
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, name='Adam').minimize(loss,
                                                                                              var_list=[var for var in
                                                                                                        tf.trainable_variables()],
                                                                                              global_step=global_step)

saver = tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()

# If a pre-trained ResNet is required, load the weights.
# This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
if init_fn is not None:
    init_fn(sess)

# Load a previous checkpoint if desired
path = args.dataset
# folder_dataset=path.split('/')[-2]
checkpoints_path = "%s/%s" % ("checkpoints", args.checkpoint)
model_checkpoint_name = checkpoints_path + "/latest_model_" + args.model + "_" + ".ckpt"
# model_checkpoint_name = "checkpoints/latest_model_" + args.model + "_" + folder_dataset + ".ckpt"
if args.continue_training and os.path.isdir(checkpoints_path):
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)
    dirs = [x[1] for x in os.walk(checkpoints_path)]
    dirs = dirs[0]
    dirs.sort()
    args.epoch_start_i = int(dirs[-1])
    print('The training starts from the last epoch {}'.format(args.epoch_start_i))

# Load the data
print("Loading the data ...")
train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(
    dataset_dir=args.dataset)

# Create a folder called results to save the files at the end of the training
results_path = os.path.join("results", args.checkpoint)
if not os.path.isdir(results_path):
    os.makedirs(os.path.join("results", args.checkpoint))

# For saving the results in a csv file:
# sheet_name= os.path.join("results",args.model + '_' + folder_dataset + '.xlsx')
# TODO: generalize the csv writting when there are more than two classes (Background + Cell)
sheet_name = os.path.join("results", args.checkpoint, args.model + '.csv')
if not args.continue_training or os.path.exists(sheet_name) == 0:
    # df = DataFrame(index=range(1), columns=['Average', 'Cell', 'Background', 'Validation precision','Validation recall','F1 score', 'IoU score', 'LR'])
    fields = 'Epoch; Average; Cell; Background; Validation precision; Validation recall; F1 score; IoU score; Average loss; LR'
    with open(sheet_name, 'w') as file_:
        file_.write(fields)
        file_.write("\n")

# initialize values to avoid any error.
avg_score = []
class_avg_scores = [0, 0]
avg_precision = []
avg_recall = []
avg_f1 = []
avg_iou = []

# else:
#     import pandas as pd
#     df=pd.read_excel(sheet_name)

print("\n***** Begin training *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", num_classes)

print("Data Augmentation:")
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tBrightness Alteration -->", args.brightness)
print("\tRotation -->", args.rotation)
print("")

avg_loss_per_epoch = []
avg_scores_per_epoch = []
avg_iou_per_epoch = []
# Create a mark to store the pots without removing previous ones:
# plt_mark = time.ctime()
plt_mark = random.randint(0, 10000)
# Define the x-limtis for the plots.
if args.epoch_start_i % args.validation_step == 0:
    e0 = args.epoch_start_i
else:
    e0 = args.epoch_start_i + args.validation_step - (args.epoch_start_i % args.validation_step)

# Which validation images do we want
val_indices = []
num_vals = min(args.num_val_images, len(val_input_names))

if args.random_val == 1:
    # Set random seed to make sure models are validated on the same validation images.
    # So you can compare the results of different models more intuitively.
    # Note: this only works if the entire images are validated rather than random
    # patches of them
    random.seed(16)
    val_indices = random.sample(range(0, len(val_input_names)), num_vals)
else:
    val_indices = np.arange(num_vals)

# Do the training here
for epoch in range(args.epoch_start_i, args.num_epochs):

    current_losses = []

    cnt = 0

    # Equivalent to shuffling
    id_list = np.random.permutation(len(train_input_names))

    num_iters = int(np.floor(len(id_list) / args.batch_size))
    st = time.time()
    epoch_st = time.time()
    for i in range(num_iters):
        # st=time.time()

        input_image_batch = []
        output_image_batch = []

        # Collect a batch of images
        for j in range(args.batch_size):
            index = i * args.batch_size + j
            id = id_list[index]
            input_image = utils.load_image(train_input_names[id])

            output_image = utils.load_image(train_output_names[id])

            with tf.device('/cpu:0'):
                input_image, output_image = data_augmentation(input_image, output_image)

                if len(input_image.shape) < 3:
                    input_image = input_image.reshape((input_image.shape[0], input_image.shape[1], 1))

                # Prep the data. Make sure the labels are in one-hot format
                # Our images are uint16
                # input_image = np.float32(input_image) / 255.0
                input_image = np.float32(input_image) / (2 ** (16) - 1)

                # Make output binary as our masks are instance masks
                if np.max(output_image) > np.max(label_values):
                    output_image[output_image > 0] = 1

                output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))

                input_image_batch.append(np.expand_dims(input_image, axis=0))
                output_image_batch.append(np.expand_dims(output_image, axis=0))

        if args.batch_size == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1), axis=0)
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1), axis=0)

        # Do the training
        _, current = sess.run([opt, loss], feed_dict={net_input: input_image_batch, net_output: output_image_batch})
        current_losses.append(current)
        cnt = cnt + args.batch_size
        if cnt % 20 == 0:
            string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f" % (
            epoch, cnt, current, time.time() - st)
            utils.LOG(string_print)
            st = time.time()

    mean_loss = np.mean(current_losses)
    avg_loss_per_epoch.append(mean_loss)

    # Create directories if needed
    # Save latest checkpoint to same file name
    print("Saving latest checkpoint")
    saver.save(sess, model_checkpoint_name)

    fig2, ax2 = plt.subplots(figsize=(11, 8))

    ax2.plot(range(args.epoch_start_i, epoch + 1), avg_loss_per_epoch)
    ax2.set_title("Average loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")

    plt.savefig('results/' + args.checkpoint + '/loss_vs_epochs-' + np.str(plt_mark) + '.png')

    plt.clf()

    if len(val_indices) != 0 and epoch % args.checkpoint_step == 0:
        # Create directories if needed
        if not os.path.isdir("%s/%04d" % (checkpoints_path, epoch)):
            os.makedirs("%s/%04d" % (checkpoints_path, epoch))
        print("Saving checkpoint for this epoch")
        saver.save(sess, "%s/%04d/model.ckpt" % (checkpoints_path, epoch))

    if epoch % args.validation_step == 0:
        # Create directories if needed
        if not os.path.isdir("%s/%04d" % (checkpoints_path, epoch)):
            os.makedirs("%s/%04d" % (checkpoints_path, epoch))
        print("Performing validation")
        target = open("%s/%04d/val_scores.csv" % (checkpoints_path, epoch), 'w')
        target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))

        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []

        # Do the validation on a small set of validation images
        for ind in val_indices:

            # input_image = np.expand_dims(np.float32(utils.load_image(val_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0

            input_image = np.float32(utils.load_image(val_input_names[ind])) / (2 ** (16) - 1)
            gt = utils.load_image(val_output_names[ind])
            if np.max(gt) > np.max(label_values):
                gt[gt > 0] = 1
            # crop a patch sampling cell bodies with a sampling pdf (cell==1 has weight 10000 and cell == 0 has weight 1)
            ## If random pathces are desired, uncomment next line
            # input_image, gt = utils.random_crop(input_image, gt, args.crop_height, args.crop_width)
            # The size of input images has to be module of 32 to have unvariable size in the output.
            offset = input_image.shape[0] % 32
            input_image = input_image[offset:]
            gt = gt[offset:]

            offset = input_image.shape[1] % 32
            input_image = input_image[:, offset:]
            gt = gt[:, offset:]

            # create an input of shape (1,:,:,:) for the network.
            if len(input_image.shape) == 2:
                input_image = input_image.reshape((input_image.shape[0], input_image.shape[1], 1))
            input_image = np.expand_dims(input_image, axis=0)

            # gt = utils.load_image(val_output_names[ind])[:args.crop_height, :args.crop_width]
            gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

            # st = time.time()
            # Process the patch
            output_image = sess.run(network, feed_dict={net_input: input_image})

            output_image = np.array(output_image[0, :, :, :])
            output_image = helpers.reverse_one_hot(output_image)

            # Calculate the accuracy measures of the obtained result.
            accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt,
                                                                                         num_classes=num_classes)

            # Store all accuracy values for this shot.
            file_name = utils.filepath_to_name(val_input_names[ind])
            target.write("%s, %f, %f, %f, %f, %f" % (file_name, accuracy, prec, rec, f1, iou))
            for item in class_accuracies:
                target.write(", %f" % (item))
            target.write("\n")

            scores_list.append(accuracy)
            class_scores_list.append(class_accuracies)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            iou_list.append(iou)

            # Store shots during the evaluation of the validation
            if args.stor_val:

                out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

                gt = helpers.colour_code_segmentation(gt, label_values)

                file_name = os.path.basename(val_input_names[ind])
                file_name = os.path.splitext(file_name)[0]
                if out_vis_image.shape[-1] == 3:
                    cv2.imwrite("%s/%04d/%s_pred.png" % (checkpoints_path, epoch, file_name),
                                cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
                    cv2.imwrite("%s/%04d/%s_gt.png" % (checkpoints_path, epoch, file_name),
                                cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite("%s/%04d/%s_pred.tif" % (checkpoints_path, epoch, file_name),
                                np.uint8(out_vis_image[:, :, 0]))
                    cv2.imwrite("%s/%04d/%s_gt.tif" % (checkpoints_path, epoch, file_name), np.uint8(gt))
                cv2.imwrite("%s/%04d/%s_input.tif" % (checkpoints_path, epoch, file_name), input_image[0, :, :, 0])

        target.close()

        avg_score = np.mean(scores_list)
        class_avg_scores = np.mean(class_scores_list, axis=0)
        avg_scores_per_epoch.append(avg_score)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)
        avg_iou_per_epoch.append(avg_iou)

        print("\nAverage validation accuracy for epoch # %04d = %f" % (epoch, avg_score))
        print("Average per class validation accuracies for epoch # %04d:" % (epoch))
        for index, item in enumerate(class_avg_scores):
            print("%s = %f" % (class_names_list[index], item))
        print("Validation precision = ", avg_precision)
        print("Validation recall = ", avg_recall)
        print("Validation F1 score = ", avg_f1)
        print("Validation IoU score = ", avg_iou)

        # create a plot with all the results
        fig1, ax1 = plt.subplots(figsize=(11, 8))

        ax1.plot(range(e0, epoch + 1, args.validation_step), avg_scores_per_epoch)
        ax1.set_title("Average validation accuracy vs epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Avg. val. accuracy")

        plt.savefig('results/' + args.checkpoint + '/accuracy_vs_epochs-' + np.str(plt_mark) + '.png')

        plt.clf()

        fig3, ax3 = plt.subplots(figsize=(11, 8))

        ax3.plot(range(e0, epoch + 1, args.validation_step), avg_iou_per_epoch)
        # plt.xticks(np.arange(args.epoch_start_i, epoch+1, step=10))
        ax3.set_title("Average IoU vs epochs")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Current IoU")

        plt.savefig('results/' + args.checkpoint + '/iou_vs_epochs-' + np.str(plt_mark) + '.png')

    epoch_time = time.time() - epoch_st
    remain_time = epoch_time * (args.num_epochs - 1 - epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s != 0:
        train_time = "Remaining training time = %d hours %d minutes %d seconds\n" % (h, m, s)
    else:
        train_time = "Remaining training time : Training completed.\n"
    utils.LOG(train_time)
    scores_list = []

    # To save the data in an excel file
    with open(sheet_name, mode='a') as file_:
        file_.write("{};{};{};{};{};{};{};{};{};{}".format(epoch, avg_score, class_avg_scores[0], class_avg_scores[1],
                                                           avg_precision, avg_recall, avg_f1, avg_iou, mean_loss,
                                                           args.learning_rate))
        file_.write("\n")