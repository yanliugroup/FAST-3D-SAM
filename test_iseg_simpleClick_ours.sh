#method_name="tri_attn_loraOnly"
method_name="tri_attn_loraAdapter_pEncodeS_miniDe"
#method_name="baidu_new"
crop_size=128
dataset_name="['lung','lung2','Lung421','pancreas','kits23','hepatic']"
#dataset_name="['lung_hospital']"
#dataset_name="['lung_pat']"
#dataset_name="['colon']"

#dataset_name="['lung','lung2','Lung421']"
#dataset_name="['pancreas']"
#dataset_name="['kits23']"
#dataset_name="['hepatic']"

load_weight="original"
#load_weight="medsam"
input_image_size=256
batch_size=2
learning_rate=0.0002
epoch=100

# exp1 SAM prompt encoder
#python test_iseg_lora_incep_simpleClick.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight --pretrained  

python test_iseg_lora_incep_simpleClick_latest_0516.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight --pretrained  