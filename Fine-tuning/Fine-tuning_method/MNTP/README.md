# MNTP

To train a model with Masked Next Token Prediction (MNTP), you can use the `run_mntp.py` script. Below is an example of how to start training:

```
python experiments/run_mntp.py \
    MNTP_codeGemma.json
```

For different models, you can modify the `"model_name_or_path"` and `"output_dir"` fields in the JSON configuration file. This MNTP training changes the model's attention mechanism from causal to bidirectional. SimCSE and SupCon training can only be performed after completing the MNTP training.
