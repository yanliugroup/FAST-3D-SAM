method_name="tri_attn_loraAdapter_pEncodeS_miniDe"
crop_size=128
#dataset_name="['colon']"
#dataset_name="['liver']"
#dataset_name="['lung_hospital']"
#dataset_name="['hippo']"
#dataset_name="['spleen']"
dataset_name="['brain_torch']"
load_weight="original"
input_image_size=256
batch_size=2
learning_rate=0.0002
#0.0002
epoch=100
num_click=100
save='no'
use_ft='yes'

python _iseg_test_by_given_clicks.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --save_result $save --use_ft_weight $use_ft --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight --num_click 50 --pretrained  