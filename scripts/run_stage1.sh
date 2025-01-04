#!/bin/bash

CONFIG=$1

python distillation/stage1_train_filters.py --config $CONFIG
