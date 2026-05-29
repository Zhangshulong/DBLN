# bash ./DBLN.sh

for dataPath in AtrialFibrillation BasicMotions CharacterTrajectories Cricket Epilepsy ERing 
do
    python -u main.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./data \
    --datatype UEA \
    --dataset $dataPath \
    --model $model_name \
    --e_layers 3 \
    --batch_size 32 \
    --d_model 128 \
    --lr 1e-4 \
    --epoch 200 \
    --patience 50 \
    --decomp 1 \
    --hvmamba 1 \
    --fuse 1 \
    --d_state 16 \
    --d_conv 4 \
    --expand 2 \
    --seed 42 \
    --use_patch_dict 1 \
    --checkpoints ./exp/checkpoints \
    --results ./exp/results \
    --device cuda:0
done

