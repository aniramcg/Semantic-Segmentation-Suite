pip install scipy==1.1.0
python train4drive.py --dataset "data" --num_epochs '1500' --learning_rate '0.001' --batch_size '3' --continue_training True --checkpoint 'constante_lr_v01' --reduce_lr 'False'
