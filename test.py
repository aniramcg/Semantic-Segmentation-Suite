import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np

from utils import utils, helpers
from builders import model_builder

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default='checkpoints/Mon May 18 15:53:51 2020/latest_model_MobileUNet_.ckpt', required=False, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default="MobileUNet", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--dataset', type=str, default="/content/gdrive/My Drive/TFG/TFG MARINA CALZADA/clean_data/data4training/SEG/", required=False, help='The dataset you are using')
args = parser.parse_args()

# Create directories if needed
if not os.path.isdir("%s"%("Test")):
        os.makedirs("%s"%("Test"))

sheet_name= os.path.join("Test",args.model + '.csv')
if os.path.exists(sheet_name)==0:
    # df = DataFrame(index=range(1), columns=['Average', 'Cell', 'Background', 'Validation precision','Validation recall','F1 score', 'IoU score', 'LR'])
    fields = 'Average; Cell; Background; Validation precision; Validation recall; F1 score; IoU score'
    with open(sheet_name, 'w') as file_:
        file_.write(fields)
        file_.write("\n")

# Get the names of the classes so we can record the evaluation results
print("Retrieving dataset information ...")
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,1]) # We work with intensities not RGB images.
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network, _ = model_builder.build_model(args.model, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=False)

sess.run(tf.global_variables_initializer())
model_checkpoint_name=args.checkpoint_path
print('Loading model checkpoint weights ...')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, model_checkpoint_name)

# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)



target=open("%s/test_scores.csv"%("Test"),'w')
target.write("test_name, test_accuracy, precision, recall, f1 score, mean iou %s\n" % (class_names_string))
scores_list = []
class_scores_list = []
precision_list = []
recall_list = []
f1_list = []
iou_list = []
run_times_list = []

# Run testing on ALL test images
for ind in range(len(test_input_names)):
    sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(test_input_names)))
    sys.stdout.flush()

    input_image = np.expand_dims(np.float32(utils.load_image(test_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/ (2**(16)-1)
    # if len(input_image.shape) < 3:
    input_image = input_image.reshape((1,input_image.shape[1], input_image.shape[2], 1))
    gt = utils.load_image(test_output_names[ind])[:args.crop_height, :args.crop_width]
    gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))
   
    
    st = time.time()
    output_image = sess.run(network,feed_dict={net_input:input_image})

    run_times_list.append(time.time()-st)

    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)
    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

    accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)

    file_name = utils.filepath_to_name(test_input_names[ind])
    target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
    for item in class_accuracies:
        target.write(", %f"%(item))
    target.write("\n")

    scores_list.append(accuracy)
    class_scores_list.append(class_accuracies)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)
    iou_list.append(iou)
    
    gt = helpers.colour_code_segmentation(gt, label_values)
cv2.imwrite("%s/%04d/%s_pred.tif" % (checkpoints_path, epoch, file_name),np.uint8(out_vis_image[:,:,0]))

    cv2.imwrite("%s/%s_pred.tif"%("Test", file_name),np.uint8(out_vis_image[:,:,0]))
    cv2.imwrite("%s/%s_gt.tif"%("Test", file_name), np.uint8(gt))


target.close()

avg_score = np.mean(scores_list)
class_avg_scores = np.mean(class_scores_list, axis=0)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_iou = np.mean(iou_list)
avg_time = np.mean(run_times_list)
print("Average test accuracy = ", avg_score)
print("Average per class test accuracies = \n")
for index, item in enumerate(class_avg_scores):
    print("%s = %f" % (class_names_list[index], item))
print("Average precision = ", avg_precision)
print("Average recall = ", avg_recall)
print("Average F1 score = ", avg_f1)
print("Average mean IoU score = ", avg_iou)
print("Average run time = ", avg_time)

with open(sheet_name, mode='a') as file_:
        file_.write("{};{};{};{};{};{};{}".format(avg_score, class_avg_scores[0], class_avg_scores[1],
                                         avg_precision, avg_recall, avg_f1, avg_iou))
        file_.write("\n")