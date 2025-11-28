#method_name="unetr"
#method_name="unetr++"
#method_name="swin_unetr"
#method_name="swin_unetr2"
method_name="3d_uxnet"
#method_name="DAF3D"

crop_size=128
#dataset_name="['lung','lung2','Lung421','pancreas','kits23','hepatic']"
dataset_name="['lung_hospital']"
#dataset_name="['lung_pat']"
#dataset_name="['liver']"
#dataset_name="['spleen']"
#dataset_name="['lung','lung2','Lung421']"
batch_size=1
learning_rate=0.001
epoch=100

# swin_unetr / 3d_uxnet
#python test_seg_traditional_sotas_4datasets.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_1click_256/ 
# DAF3D / unetr++
python test_seg_traditional_sotas_0516.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_NoClick_-weight_norm/ 