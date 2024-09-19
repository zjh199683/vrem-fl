# Go inside desired folder
cd ../

cd system_configuration/
python generate_cfg_file.py

cd ../
python run_experiment.py \
  apolloscape \
  optimal \
  --bz 4 \
  --optimizer sgd \
  --local_lr 0.001 \
  --lr_scheduler constant \
  --min_local_steps 1 \
  --max_local_steps 50 \
  --device cuda \
  --log_freq 1 \
  --verbose 1 \
  --logs_save_path logs/logs_optimal_apolloscape \
  --n_clients 5 \
  --sim_len 780 \
  --cfg_file_path system_configuration/system_cfg.json \
  --seed 0
