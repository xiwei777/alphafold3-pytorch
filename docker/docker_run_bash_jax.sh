docker run -itd \
    --volume $AF3_HOME/af_input:/root/af_input \
    --volume $AF3_HOME/af_output:/root/af_output \
    --volume $AF3_HOME/models:/root/models \
    --volume $AF3_HOME/public_databases:/root/public_databases \
    --ipc=host\
    -p 9499:22 -p 9500-9550:9500-9550 \
    --gpus all \
    --name open-sora-dokcer\
    open-sora \
    /bin/bash
# python run_alphafold.py \
# --json_path=/root/af_input/fold_input.json \
# --model_dir=/root/models \
# --output_dir=/root/af_output