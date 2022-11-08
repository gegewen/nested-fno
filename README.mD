## Make dataloader
- step 1: download meta data from google drive and put them into `Nested_FNO/ECLIPSE/meta_data`
- step 2: run following code to convert `.npy` file into `.pt` files in `dataset` folder
```
cd data_config
bash file_config.sh
cd ..
``` 
- step 3: run `python3 save_data_loader.py` to create `DATA_LOADER_DICT.pth`

## Evaluate with pre-trained weights
- step 1: download `FNO4D-GLOBAL-DP.pt`, `FNO4D-LGR1..4-DP.pt`, `FNO4D-LGR1..4-SG.pt` models and place in `pre_trained_models`
- step 2: run `eval_sequential_prediction_dp.ipynb` and `eval_sequential_prediction_sg.ipynb` to evaluate

## Training procedure
- step 1: train each models seperately using the following code. Each model requires an NVIDIA A100 GPU.
```
python3 train_FNO4D_DP_GLOBAL.py
python3 train_FNO4D_DP_LGR.py LGR1
python3 train_FNO4D_DP_LGR.py LGR2
python3 train_FNO4D_DP_LGR.py LGR3
python3 train_FNO4D_DP_LGR.py LGR4
python3 train_FNO4D_SG_LGR.py LGR1
python3 train_FNO4D_SG_LGR.py LGR2
python3 train_FNO4D_SG_LGR.py LGR3
python3 train_FNO4D_SG_LGR.py LGR4
```
- step 2: monitor training and validation loss with tensorboard
```
tensorboard --logdir=logs --port=6007 --host=sh03-14n13
```

## Finetune procedure
In the paper, we finetuned `dP_LGR1`, `dP_LGR4`, `SG_LGR1`, `SG_LGR1` models with a random instance of pre-generated error. Use `finetune_FNO4D_DP_LGR.ipynb` as an example.