# python trainer_vanilla.py \
# --proj_name Vanilla \
# --model_type Vanilla \
# --data_type FashionMNIST \
# --output_dir ./Results/Vanilla/ \
# --gpu_devices 1 \
# --data_dir ./dataset

# python trainer_seg_on_carvana.py \
# --proj_name SegOnCarvana \
# --model_type UNet \
# --data_type Carvana \
# --output_dir ./Results/Carvana/ \
# --gpu_devices 0 \
# --data_dir /mnt/data4/Dataset/Carvana/

# python eval_seg_on_carvana.py \
# --model_path ./Results/Carvana/1664021850/UNet_best_ckpt.pth \
# --save_path ./test_viz/ \
# --data_dir /mnt/data4/Dataset/Carvana \
# --model_type UNet \
# --data_type Carvana \
# --viz True