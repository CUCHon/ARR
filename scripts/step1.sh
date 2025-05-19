





# torchrun \
#   --nproc_per_node=7 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=localhost:29503 \
#   src/generate_scores.py \
#     --input_path data/alpaca_data.json \
#     --output_path data/alpaca_data-step1.json \
#     --sbert_model all-MiniLM-L6-v2 \
#     --cross_model cross-encoder/stsb-distilroberta-base \
#     --quality_model hkust-nlp/deita-quality-scorer \
#     --complexity_model hkust-nlp/deita-complexity-scorer \
#     --max_len 2048 \
#     --top_k 5 \
#     --batch_size 4

# torchrun \
#   --nproc_per_node=8 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=localhost:29502 \
#   src/generate_scores.py \
#     --input_path data/wizardv1-filtered_with_input.json \
#     --output_path data/wizardv1-filtered_with_input-step1.json \
#     --sbert_model all-MiniLM-L6-v2 \
#     --cross_model cross-encoder/stsb-distilroberta-base \
#     --quality_model hkust-nlp/deita-quality-scorer \
#     --complexity_model hkust-nlp/deita-complexity-scorer \
#     --max_len 2048 \
#     --top_k 5 \
#     --batch_size 8

# torchrun \
#   --nproc_per_node=3 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=localhost:29501 \
#   src/generate_scores.py \
#     --input_path data/alpaca_gpt4_data_en.json \
#     --output_path data/alpaca_gpt4_data_en-step1.json \
#     --sbert_model all-MiniLM-L6-v2 \
#     --cross_model cross-encoder/stsb-distilroberta-base \
#     --quality_model hkust-nlp/deita-quality-scorer \
#     --complexity_model hkust-nlp/deita-complexity-scorer \
#     --max_len 2048 \
#     --top_k 5 \
#     --batch_size 8
