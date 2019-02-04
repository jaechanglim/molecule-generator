# Tensorflow implementation of Generating Focussed Molecule Libraries for Drug Discovery with Recurrent Neural Networks
https://arxiv.org/abs/1701.01329

1. You need a smiles (smiles.txt) file for pretraining the model. 
2. Train the model using following command
```
python -u train.py --smiles_data=smiles.txt --vocab_from=smiles.txt --save_dir=./save --lr=1e-4
```
3. Retrain the model using pretrained model with low learning rate
```
python -u train.py --smiles_data=egfr_smiles.txt --vocab_from=smiles.txt --save_dir=./save_egfr --lr=1e-5 --num_epochs=10  --pretrained=./save/model_30.ckpt-30
```
4. Generate molecules. The result will be written in result.txt
```
python sample.py --vocab_from=smiles.txt --save_file=save_egfr/model_9.ckpt-9 --result_filename=result.txt
```
