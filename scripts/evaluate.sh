#!/bin/bash

# Usage: bash evaluate.sh <task_type> <model_path> <task_name>

TASK_TYPE=$1
MODEL_PATH=$2
TASK_NAME=$3

if [ "$TASK_TYPE" == "language_modeling" ]; then
    python experiments/language_modeling/evaluate_gpt2_student.py --model_path $MODEL_PATH --task_name "wikitext"
elif [ "$TASK_TYPE" == "nlu" ]; then
    python experiments/nlu_tasks/evaluate_deberta_student.py --model_path $MODEL_PATH --task_name $TASK_NAME
else
    echo "Unsupported task type. Use 'language_modeling' or 'nlu'."
fi
