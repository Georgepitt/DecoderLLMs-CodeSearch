## SimCSE

To train using SimCSE, you can use the `run_simcse.py` script. Before running the script, make sure to download the dataset and save it according to the specified format. Additionally, update the `"peft_model_name_or_path"` field in the `SimCSE_codegemma.json` file to point to the model path obtained after MNTP training.

### SimCSE Training

```
cd decoder-only-code-search/Fine-tuning/Fine-tuning_method/SimCSE

python run_simcse.py \
simce_codegemma.json
```

For different models, you can modify the `"model_name_or_path"`, `"peft_model_name_or_path"`, and `"output_dir"` fields in the JSON configuration file.
