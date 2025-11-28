#method_name="baidu_new"
#method_name="lora_incep"
method_name="baidu_new"
crop_size=128
dataset_name="['lung','lung2','Lung421','pancreas','kits23','hepatic']"
load_weight="original"
input_image_size=256
batch_size=2
learning_rate=0.0004
epoch=100
max_clicks=12

# exp1 SAM prompt encoder
#python train_iseg_lora_incep_SAM_prompt.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_NoClick_sam_${load_weight}-weight_norm/ --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight


python train_iseg_lora_incep_SAM_prompt_random_click_num.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_NoClick_sam_${load_weight}-weight_norm/ --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight --max_clicks $max_clicks --pretrained  