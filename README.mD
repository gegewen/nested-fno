## Make dataloader
- step 1: download meta data from [google drive](https://drive.google.com/drive/u/1/folders/1gElIBiZW6NayuEWxgDn8cv94_4e_LF4-) and put them into `Nested_FNO/ECLIPSE/meta_data`
- step 2: run following code to convert `.npy` file into `.pt` files in `dataset` folder
```
cd data_config
bash file_config.sh
cd ..
``` 
- step 3: run `python3 save_data_loader.py` to create `DATA_LOADER_DICT.pth`

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
As discussed in the paper, we finetuned `dP_LGR1`, `dP_LGR4`, `SG_LGR1`, `SG_LGR1` models with a random instance of pre-generated error. 