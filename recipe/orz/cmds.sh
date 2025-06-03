CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash /home/aiscuser/verl/recipe/orz/run_orz_llama_8b_rpp.sh False False

CUDA_VISIBLE_DEVICES=4,5,6,7 \
bash /home/aiscuser/verl/recipe/orz/run_orz_llama_8b_rpp.sh False False

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash /home/aiscuser/verl/recipe/orz/run_orz_llama_8b_rpp.sh True False

CUDA_VISIBLE_DEVICES=4,5,6,7 \
bash /home/aiscuser/verl/recipe/orz/run_orz_llama_8b_rpp.sh True True