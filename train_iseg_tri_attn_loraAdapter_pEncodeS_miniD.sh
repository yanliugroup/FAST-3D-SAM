method_name="tri_attn_loraAdapter_pEncodeS_miniDe"
crop_size=128
#dataset_name="['lung','lung2','Lung421','pancreas','kits23','hepatic']"
#dataset_name="['liver','spleen','hippo','colon']"
#dataset_name="['liver']"
#dataset_name="['lung_hospital']"
#dataset_name="['lung_pat']"
#dataset_name="['hippo']"
#dataset_name="['colon']"
#dataset_name="['spleen']"
dataset_name="['brain_torch']"
load_weight="original"
input_image_size=256
batch_size=2
#learning_rate=0.001
learning_rate=0.0002
#0.0002
epoch=50

# exp1 SAM prompt encoder
python train_iseg_tri_attn_lora_adapter_pEncodeS_miniDe.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight --pretrained  


# python train_iseg_tri_attn_lora_adapter_pEncodeS_miniDe.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight
