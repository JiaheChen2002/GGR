today=`date +%T`
echo $today
gpu_id=$1
trainer=$2

if [[ "${trainer}" == *"supervised"* ]] || [[ "${trainer}" == *"mixmatch"* ]] || [[ "${trainer}" == *"fixmatch"* ]] || \
   [[ "${trainer}" == *"comatch"* ]] || [[ "${trainer}" == *"simmatch"* ]] || [[ "${trainer}" == *"softmatch"* ]]; then
    ssl_type=classic_cv
else
    ssl_type=openset_cv
fi

dataset=cifar100
num_classes=50
for seed in 0 1 2 3
do
    for n_labels in 5 10 25
    do
        num_labels=$((n_labels * num_classes))
        config_path=config/${ssl_type}/${trainer}/${trainer}_${dataset}_${num_classes}_${num_labels}_${seed}.yaml
        if [ -f "${config_path}" ]; then
            CUDA_VISIBLE_DEVICES=$gpu_id python train.py --c "${config_path}"
        else
            echo "skip missing config: ${config_path}"
        fi
    done
done
