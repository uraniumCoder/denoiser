mkdir -p models/pruned

## declare an array variable
declare -a arr=("99" "95" "90" "80" "70" "60" "50" "40" "30" "20" "10")

## now loop through the above array
python -m denoiser.prune --dns48 --uniform --p 1 --output models/pruned/48_p100.t
python -m denoiser.prune --dns64 --uniform --p 1 --output models/pruned/64_p100.t

for i in "${arr[@]}"
do
   python -m denoiser.prune --dns48 --uniform --p 0.$i --output models/pruned/48_p$i.t
   python -m denoiser.prune --dns64 --uniform --p 0.$i --output models/pruned/64_p$i.t
done

