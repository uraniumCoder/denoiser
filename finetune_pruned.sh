# declare -a arr=("90" "80" "70" "60" "50")
declare -a arr=("99" "95")

for i in "${arr[@]}"
do
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./train.py finetune=true model_path=/mnt/cfs/home/wzhao6/callapp/denoiser/models/pruned/48_p$i.t ddp=1 demucs.prune_ratio=0.$i
done
