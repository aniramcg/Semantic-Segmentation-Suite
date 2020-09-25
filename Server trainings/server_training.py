import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="/data/", help='Dataset you are using.') #NEED TO PUT A FULL DIRECTORY, IT WON'T WORK ONLY PUTTING THE FOLDER
parser.add_argument('--num_epochs', type=str, default='1', help='Number of epochs to train for')
parser.add_argument('--learning_rate', type=str, default='0.01', help='Initial learning rate')
parser.add_argument('--batch_size', type=str, default='1', help='Number of images in each batch')
args = parser.parse_args()

try:
	subprocess.check_output(['pip', 'install', 'scipy==1.1.0'])
	subprocess.check_output(['python', 'train4drive.py', '--dataset', args.dataset, '--num_epochs', args.num_epochs, '--learning_rate', args.learning_rate, '--batch_size', args.batch_size])
except Exception as e:
	print(e)
	pass

