# train
## Pretrain
nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 4 > 1.log 2>&1 &
nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 8 > 2.log 2>&1 &
nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 > 3.log 2>&1 &
nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 32 > 4.log 2>&1 &
nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 64 > 5.log 2>&1 &

nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet --lr 1e-4 --batch_size 16 > 6.log 2>&1 &
nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet --lr 1e-5 --batch_size 16 > 7.log 2>&1 &
nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet --lr 1e-7 --batch_size 16 > 8.log 2>&1 &

nohup python train.py --train_data OS-ESB-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 > 9.log 2>&1 &
nohup python train.py --train_data OS-NTU-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 > 10.log 2>&1 &
nohup python train.py --train_data OS-MN40-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 > 11.log 2>&1 &

## dis_Pre
nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 4 > 12.log 2>&1 &
nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 8 > 13.log 2>&1 &
nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 > 14.log 2>&1 &
nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 32 > 15.log 2>&1 &
nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 64 > 16.log 2>&1 &


nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-4 --batch_size 16 > 17.log 2>&1 &
nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-5 --batch_size 16 > 18.log 2>&1 &
nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-7 --batch_size 16 > 19.log 2>&1 &

nohup python train.py --train_data OS-ESB-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 > 20.log 2>&1 &
nohup python train.py --train_data OS-NTU-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 > 21.log 2>&1 &
nohup python train.py --train_data OS-MN40-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 > 22.log 2>&1 &


## dis_Pre_RLT
nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss > 23.log 2>&1 &
nohup python train.py --train_data OS-ESB-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss  > 24.log 2>&1 &
nohup python train.py --train_data OS-NTU-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss  > 25.log 2>&1 &
nohup python train.py --train_data OS-MN40-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss  > 26.log 2>&1 &

## dis_Pre_Com
nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss CombineLoss > 27.log 2>&1 &
nohup python train.py --train_data OS-ESB-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss CombineLoss  > 28.log 2>&1 &
nohup python train.py --train_data OS-NTU-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss CombineLoss  > 29.log 2>&1 &
nohup python train.py --train_data OS-MN40-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss CombineLoss  > 30.log 2>&1 &

## train_on_DS
nohup python train.py --train_data MN40-DS --model_name MV_AlexNet --lr 1e-6 --batch_size 16 > 31.log 2>&1 &
nohup python train.py --train_data MN40-DS --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 > 32.log 2>&1 &
nohup python train.py --train_data MN40-DS --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss  > 33.log 2>&1 &

# feature
## Pretrain
nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core  --model_name MV_AlexNet --lr 1e-6 --batch_size 4 > f_1.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 8 > f_2.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 > f_3.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 32 > f_4.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 64 > f_5.log 2>&1 &

nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-4 --batch_size 16 > f_6.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-5 --batch_size 16 > f_7.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-7 --batch_size 16 > f_8.log 2>&1 &

nohup python feature_extraction.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 > f_9.log 2>&1 &
nohup python feature_extraction.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 > f_10.log 2>&1 &
nohup python feature_extraction.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 > f_11.log 2>&1 &

## dis_Pre
nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 4 > f_12.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 8 > f_13.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 > f_14.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 32 > f_15.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 64 > f_16.log 2>&1 &


nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-4 --batch_size 16 > f_17.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-5 --batch_size 16 > f_18.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-7 --batch_size 16 > f_19.log 2>&1 &

nohup python feature_extraction.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 > f_20.log 2>&1 &
nohup python feature_extraction.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 > f_21.log 2>&1 &
nohup python feature_extraction.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 > f_22.log 2>&1 &


## dis_Pre_RLT
nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss > f_23.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss  > f_24.log 2>&1 &
nohup python feature_extraction.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss  > f_25.log 2>&1 &
nohup python feature_extraction.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss  > f_26.log 2>&1 &

## dis_Pre_COm
nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss CombineLoss > f_51.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss CombineLoss  > f_52.log 2>&1 &
nohup python feature_extraction.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss CombineLoss  > f_53.log 2>&1 &
nohup python feature_extraction.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss CombineLoss  > f_54.log 2>&1 &

## MV_CLIP_without_adpter

nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_CLIP_without_adpter > f_27.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_CLIP_without_adpter > f_28.log 2>&1 &
nohup python feature_extraction.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_CLIP_without_adpter > f_29.log 2>&1 &
nohup python feature_extraction.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_CLIP_without_adpter > f_30.log 2>&1 &

## views
nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 --num_views 4 > f_31.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 --num_views 4 > f_46.log 2>&1 &
nohup python feature_extraction.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 --num_views 4 > f_47.log 2>&1 &
nohup python feature_extraction.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 --num_views 4 > f_48.log 2>&1 &

nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16  --num_views 4 > f_32.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16  --num_views 4 > f_33.log 2>&1 &
nohup python feature_extraction.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16  --num_views 4 > f_34.log 2>&1 &
nohup python feature_extraction.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16  --num_views 4 > f_35.log 2>&1 &

nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --num_views 4 > f_36.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --num_views 4 > f_37.log 2>&1 &
nohup python feature_extraction.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --num_views 4 > f_38.log 2>&1 &
nohup python feature_extraction.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --num_views 4 > f_39.log 2>&1 &

nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_CLIP_without_adpter --num_views 4 > f_40.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_CLIP_without_adpter --num_views 4 > f_41.log 2>&1 &
nohup python feature_extraction.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_CLIP_without_adpter --num_views 4 > f_42.log 2>&1 &
nohup python feature_extraction.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_CLIP_without_adpter --num_views 4 > f_43.log 2>&1 &

nohup python feature_extraction.py --train_data OS-ABO-core --test_dataset OS-MN40-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 > f_49.log 2>&1 &
nohup python feature_extraction.py --train_data OS-MN40-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 > f_50.log 2>&1 &
nohup python feature_extraction.py --train_data OS-ABO-core  --test_dataset OS-MN40-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 > f_44.log 2>&1 &
nohup python feature_extraction.py --train_data OS-MN40-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 > f_45.log 2>&1 &

## DS DU
nohup python feature_extraction.py --train_data MN40-DS --test_dataset MN40-DS --model_name MV_AlexNet --lr 1e-6 --batch_size 16 > f_57.log 2>&1 &
nohup python feature_extraction.py --train_data MN40-DS --test_dataset MN40-DU --model_name MV_AlexNet --lr 1e-6 --batch_size 16 > f_58.log 2>&1 &
nohup python feature_extraction.py --train_data MN40-DS --test_dataset MN40-DS --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 > f_55.log 2>&1 &
nohup python feature_extraction.py --train_data MN40-DS --test_dataset MN40-DU --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 > f_56.log 2>&1 &
nohup python feature_extraction.py --train_data MN40-DS --test_dataset MN40-DS --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss > f_61.log 2>&1 &
nohup python feature_extraction.py --train_data MN40-DS --test_dataset MN40-DU --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss > f_62.log 2>&1 &
nohup python feature_extraction.py --test_dataset MN40-DU --model_name MV_CLIP_without_adpter > f_60.log 2>&1 &
nohup python feature_extraction.py --test_dataset MN40-DS --model_name MV_CLIP_without_adpter > f_59.log 2>&1 &

# eva
## MV_AlexNet
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 4 --order 1 > e_1.log 2>&1 &
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 8 --order 2 > e_2.log 2>&1 &
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 --order 3 > e_3.log 2>&1 &
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 32 --order 4 > e_4.log 2>&1 &
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 64 --order 5 > e_5.log 2>&1 &

nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-4 --batch_size 16 --order 6 > e_6.log 2>&1 &
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-5 --batch_size 16 --order 7 > e_7.log 2>&1 &
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-7 --batch_size 16 --order 8 > e_8.log 2>&1 &

nohup python evaluate.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 --order 9 > e_9.log 2>&1 &
nohup python evaluate.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 --order 10 > e_10.log 2>&1 &
nohup python evaluate.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 --order 11 > e_11.log 2>&1 &

## dis_Pre
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 4 --order 12 > e_12.log 2>&1 &
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 8 --order 13 > e_13.log 2>&1 &
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --order 14 > e_14.log 2>&1 &
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 32 --order 15 > e_15.log 2>&1 &
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 64 --order 16 > e_16.log 2>&1 &

nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-4 --batch_size 16 --order 17 > e_17.log 2>&1 &
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-5 --batch_size 16 --order 18 > e_18.log 2>&1 &
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-7 --batch_size 16 --order 19 > e_19.log 2>&1 &

nohup python evaluate.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --order 20 > e_20.log 2>&1 &
nohup python evaluate.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --order 21 > e_21.log 2>&1 &
nohup python evaluate.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --order 22 > e_22.log 2>&1 &

## dis_Pre_RLT
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --order 23 > e_23.log 2>&1 &
nohup python evaluate.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --order 24 > e_24.log 2>&1 &
nohup python evaluate.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --order 25 > e_25.log 2>&1 &
nohup python evaluate.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --order 26 > e_26.log 2>&1 &


## dis_Pre_Com
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss CombineLoss --order 51 > e_51.log 2>&1 &
nohup python evaluate.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss CombineLoss --order 52 > e_52.log 2>&1 &
nohup python evaluate.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss CombineLoss --order 53 > e_53.log 2>&1 &
nohup python evaluate.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss CombineLoss --order 54 > e_54.log 2>&1 &

## MV_CLIP_without_adpter
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_CLIP_without_adpter --order 27 > e_27.log 2>&1 &
nohup python evaluate.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_CLIP_without_adpter --order 28 > e_28.log 2>&1 &
nohup python evaluate.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_CLIP_without_adpter --order 29 > e_29.log 2>&1 &
nohup python evaluate.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_CLIP_without_adpter --order 30 > e_30.log 2>&1 &

## views
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 --num_views 4 --order 31 > e_31.log 2>&1 &
nohup python evaluate.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 --num_views 4 --order 46 > e_46.log 2>&1 &
nohup python evaluate.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 --num_views 4 --order 47 > e_47.log 2>&1 &
nohup python evaluate.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 --num_views 4 --order 48 > e_48.log 2>&1 &

nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --num_views 4 --order 32 > e_32.log 2>&1 &
nohup python evaluate.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --num_views 4 --order 33 > e_33.log 2>&1 &
nohup python evaluate.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --num_views 4 --order 34 > e_34.log 2>&1 &
nohup python evaluate.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --num_views 4 --order 35 > e_35.log 2>&1 &

nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --num_views 4 --order 36 > e_36.log 2>&1 &
nohup python evaluate.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --num_views 4 --order 37 > e_37.log 2>&1 &
nohup python evaluate.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --num_views 4 --order 38 > e_38.log 2>&1 &
nohup python evaluate.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --num_views 4 --order 39 > e_39.log 2>&1 &

nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_CLIP_without_adpter --num_views 4 --order 40 > e_40.log 2>&1 &
nohup python evaluate.py --train_data OS-ESB-core --test_dataset OS-ESB-core --model_name MV_CLIP_without_adpter --num_views 4 --order 41 > e_41.log 2>&1 &
nohup python evaluate.py --train_data OS-NTU-core --test_dataset OS-NTU-core --model_name MV_CLIP_without_adpter --num_views 4 --order 42 > e_42.log 2>&1 &
nohup python evaluate.py --train_data OS-MN40-core --test_dataset OS-MN40-core --model_name MV_CLIP_without_adpter --num_views 4 --order 43 > e_43.log 2>&1 &

nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-MN40-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 --order 49 > e_49.log 2>&1 &
nohup python evaluate.py --train_data OS-MN40-core --test_dataset OS-ABO-core --model_name MV_AlexNet --lr 1e-6 --batch_size 16 --order 50 > e_50.log 2>&1 &
nohup python evaluate.py --train_data OS-ABO-core --test_dataset OS-MN40-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --order 44 > e_44.log 2>&1 &
nohup python evaluate.py --train_data OS-MN40-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --order 45 > e_45.log 2>&1 &


## DS DU
nohup python evaluate.py --train_data MN40-DS --test_dataset MN40-DS --model_name MV_AlexNet --lr 1e-6 --batch_size 16 --order 57 > e_57.log 2>&1 &
nohup python evaluate.py --train_data MN40-DS --test_dataset MN40-DU --model_name MV_AlexNet --lr 1e-6 --batch_size 16 --order 58 > e_58.log 2>&1 &
nohup python evaluate.py --train_data MN40-DS --test_dataset MN40-DS --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --order 55 > e_55.log 2>&1 &
nohup python evaluate.py --train_data MN40-DS --test_dataset MN40-DU --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --order 56 > e_56.log 2>&1 &
nohup python feature_extraction.py --train_data MN40-DS --test_dataset MN40-DS --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --order 61 > e_61.log 2>&1 &
nohup python feature_extraction.py --train_data MN40-DS --test_dataset MN40-DU --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --order 62 > e_62.log 2>&1 &
nohup python evaluate.py --test_dataset MN40-DU --model_name MV_CLIP_without_adpter --order 60 > e_60.log 2>&1 &
nohup python evaluate.py --test_dataset MN40-DS --model_name MV_CLIP_without_adpter --order 59 > e_59.log 2>&1 &


### RalationLoss 调参
基于batch16 lr 1e-6 的AlexNet
nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --rGL 0.8 --rOI 0.5 > Adp_R_85.log 2>&1 &

nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --rGL 0.8 --rOI 1 > Adp_R_810.log 2>&1 &

nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --rGL 0.8 --rOI 0.8 > Adp_R_88.log 2>&1 &

nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --rGL 0.8 --rOI 0.2 > Adp_R_82.log 2>&1 &

nohup python train.py --train_data OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --rGL 0.8 --rOI 0 > Adp_R_80.log 2>&1 &

python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --rGL 0.8 --rOI 0.5
python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --rGL 0.8 --rOI 1
python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --rGL 0.8 --rOI 0.8
python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --rGL 0.8 --rOI 0.2
python evaluate.py --train_data OS-ABO-core --test_dataset OS-ABO-core --model_name MV_AlexNet_dis_Pre --lr 1e-6 --batch_size 16 --loss RelationDisLoss --rGL 0.8 --rOI 0
