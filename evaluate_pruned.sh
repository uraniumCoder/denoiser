declare -a arr=("99" "95" "90" "80" "70" "60" "50" "40" "30" "20" "10")

for i in "${arr[@]}"
do
    echo $i
    python -m denoiser.evaluate --model_path=models/pruned/48_p$i.t --data_dir=egs/valentini --device cuda --convert --updates 1 > results/48_p$i.txt
    python -m denoiser.evaluate --model_path=models/pruned/64_p$i.t --data_dir=egs/valentini --device cuda --convert --updates 1 > results/64_p$i.txt
done