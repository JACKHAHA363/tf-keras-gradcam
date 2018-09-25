#!/usr/bin/env bash

saved_model_cli run --dir ./gcam_models/ --tag_set serve \
    --signature_def serving_default \
    --input_exp 'input_image=np.random.rand(2,224,224,3)'