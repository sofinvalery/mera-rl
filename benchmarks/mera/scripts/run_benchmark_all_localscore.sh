#!/bin/bash

MERA_COMMON_SETUP_default="--model hf --device cuda --batch_size=1 --log_samples --seed 1234 --verbosity ERROR"
MERA_COMMON_SETUP="${MERA_COMMON_SETUP:-$MERA_COMMON_SETUP_default}"
RUHUMANEVAL_GEN_KWARGS="${RUHUMANEVAL_GEN_KWARGS:-temperature=0.6,do_sample=True}"
GENERATION_KWARGS="${GENERATION_KWARGS:-do_sample=False}"
DATASETS_CACHE_DIR="${MERA_DATASETS_CACHE:-${MERA_FOLDER}/ds_cache}"

FEWSHOTS=(
 2
 1
 0
 1
)

TASKS=(
"simplear_localscore"
"bps_localscore lcs_localscore bps_gen_localscore lcs_gen_localscore chegeka_localscore mathlogicqa_localscore mathlogicqa_gen_localscore parus_localscore rcb_localscore parus_gen_localscore rcb_gen_localscore rudetox_localscore rummlu_localscore rummlu_gen_localscore ruworldtree_localscore ruopenbookqa_localscore ruworldtree_gen_localscore ruopenbookqa_gen_localscore rumultiar_localscore use_localscore rwsd_localscore rwsd_gen_localscore mamuramu_localscore mamuramu_gen_localscore rutie_localscore rutie_gen_localscore"
"multiq_localscore rumodar_localscore"
""
)

for fewshot_idx in "${!FEWSHOTS[@]}"
do
  for cur_task in ${TASKS[$fewshot_idx]}
  do
    printf "task: %s \n" "$cur_task"
    if [[ "$cur_task" == *"ruhumaneval"* || "$cur_task" == *"rucodeeval"* ]]; then
        GEN_KWARGS=${RUHUMANEVAL_GEN_KWARGS}; else
        GEN_KWARGS=${GENERATION_KWARGS}; fi

    if test -z "${SYSTEM_PROMPT}"
    then
        HF_DATASETS_CACHE="${DATASETS_CACHE_DIR}" TOKENIZERS_PARALLELISM=false HF_DATASETS_IN_MEMORY_MAX_SIZE=23400000 \
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PYTHONPATH=$PWD \
        lm_eval --model hf --model_args "${MERA_MODEL_STRING}" --tasks $cur_task \
        --num_fewshot=${FEWSHOTS[$fewshot_idx]} --gen_kwargs="${GEN_KWARGS}" \
        --output_path="${MERA_FOLDER}" ${MERA_COMMON_SETUP} \
        --include_path=./benchmark_tasks
    else
        PROCESSED_SYSTEM=$(printf "%b" "$SYSTEM_PROMPT")
        HF_DATASETS_CACHE="${DATASETS_CACHE_DIR}" TOKENIZERS_PARALLELISM=false HF_DATASETS_IN_MEMORY_MAX_SIZE=23400000 \
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PYTHONPATH=$PWD \
        lm_eval --model hf --model_args "${MERA_MODEL_STRING}" --tasks $cur_task \
        --num_fewshot=${FEWSHOTS[$fewshot_idx]} --gen_kwargs="${GEN_KWARGS}" \
        --output_path="${MERA_FOLDER}" ${MERA_COMMON_SETUP} --system_instruction="${PROCESSED_SYSTEM}" \
        --include_path=./benchmark_tasks
    fi
  done
done
