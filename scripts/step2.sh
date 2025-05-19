


# torchrun \
#   --nproc_per_node=8 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=localhost:29505 \
#   src/ICL-influence-new.py \
#     --data_path=data/alpaca_gpt4_data_en-step1.json \
#     --save_path=data/alpaca_gpt4_data_en-step2.json \
#     --model_name_or_path=meta-llama/Llama-3.2-3B \
#     --max_length=4096







torchrun \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29507 \
  src/ICL-influence-new.py \
    --data_path=data/wizardv1-filtered_with_input-step1.json \
    --save_path=data/wizardv1-filtered_with_input-step2.json \
    --model_name_or_path=meta-llama/Llama-3.2-3B \
    --max_length=4096


# torchrun \
#   --nproc_per_node=7 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=localhost:29506 \
#   src/ICL-influence-new.py \
#     --data_path=data/alpaca_data-step1.json \
#     --save_path=data/alpaca_data-step2.json \
#     --model_name_or_path=meta-llama/Llama-3.2-3B \
#     --max_length=4096
