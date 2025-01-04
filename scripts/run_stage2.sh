#!/bin/bash

CONFIG=$1

python distillation/stage2_train_student.py --config $CONFIG
