# SupCon Training

Before starting the SupCon training, ensure the following prerequisites are met:

1. Download the required dataset and save it in the specified format.

2. The model intended for SupCon training must have already undergone MNTP training.

3. Update your libraries to support CSN data by running the following commands:

   ```
   cd decoder-only-code-search/Fine-tuning/Fine-tuning_method/SupCon/CSN
   
   chmod +x update.sh
   
   ./update.sh
   ```

### SupCon Training on CSN

Before training the model on the CSN dataset, update the `"peft_model_name_or_path"` field in `SupCon_CSN_CodeGemma.json` to the model path obtained after MNTP training. You can then start the SupCon training on the CSN dataset using the following command:

```
cd ..

torchrun \
    run_SupCon.py \
    SupCon_CSN_CodeGemma.json
```

### SupCon Training on E5

Similarly, before training the model on the E5 dataset, update the `"peft_model_name_or_path"` field in `SupCon_E5_CodeGemma.json` to the model path obtained after MNTP training. Start the SupCon training on the E5 dataset using the following command:

```
torchrun \
    run_SupCon.py \
    SupCon_E5_CodeGemma.json
```

For both types of training, you can modify the `"model_name_or_path"`, `"peft_model_name_or_path"`, and `"output_dir"` fields in the JSON configuration file to accommodate different models.
