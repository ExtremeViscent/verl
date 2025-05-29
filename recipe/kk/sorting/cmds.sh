CUDA_VISIBLE_DEVICES=0,1,2,3 \
SORT_MODE=scatter \
SORT_METRIC=reward \
bash /home/aiscuser/verl/recipe/kk/sorting/run_kk_llama_7b_rpp.sh False False

CUDA_VISIBLE_DEVICES=4,5,6,7 \
SORT_MODE=cluster \
SORT_METRIC=reward \
bash /home/aiscuser/verl/recipe/kk/sorting/run_kk_llama_7b_rpp.sh False False

CUDA_VISIBLE_DEVICES=0,1,2,3 \
SORT_MODE=scatter \
SORT_METRIC=length \
bash /home/aiscuser/verl/recipe/kk/sorting/run_kk_llama_7b_rpp.sh False False

CUDA_VISIBLE_DEVICES=4,5,6,7 \
SORT_MODE=cluster \
SORT_METRIC=length \
bash /home/aiscuser/verl/recipe/kk/sorting/run_kk_llama_7b_rpp.sh False False

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash /home/aiscuser/verl/recipe/dapo/run_dapo_qwen2.5_7b_dbg.sh False False