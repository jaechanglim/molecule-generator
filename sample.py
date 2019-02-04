from model import CVAE
from utils import *
import numpy as np
import os
import tensorflow as tf
import time
import argparse
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
parser.add_argument('--num_iteration', help='num_iteration', default=10)
parser.add_argument('--latent_size', help='latent_size', type=int, default=200)
parser.add_argument('--unit_size', help='unit_size of rnn cell', type=int, default=512)
parser.add_argument('--n_rnn_layer', help='number of rnn layer', type=int, default=3)
parser.add_argument('--seq_length', help='max_seq_length', type=int, default=120)
parser.add_argument('--mean', help='mean of VAE', type=float, default=0.0)
parser.add_argument('--stddev', help='stddev of VAE', type=float, default=1.0)
parser.add_argument('--num_prop', help='number of propertoes', type=int, default=3)
parser.add_argument('--save_file', help='save file', type=str)
parser.add_argument('--result_filename', help='name of result filename', type=str, default='result.txt')
parser.add_argument('--smiles_data', help='smiles data', type=str)
parser.add_argument('--vocab_from', help='the file where vocab is extracted from', type=str)
parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)
args = parser.parse_args()

#extact vocab and char
char, vocab = extract_vocab(args.vocab_from, args.seq_length)
vocab_size = len(char)

#model and restore model parapmeters
model = CVAE(vocab_size,
             args
             )
model.restore(args.save_file)

print ('Number of parameters : ', np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))

#target property to numpy array
start_codon = np.array([np.array(list(map(vocab.get, 'X')))for _ in range(args.batch_size)])

#generate smiles
smiles = []
for _ in range(args.num_iteration):
    generated = model.sample(start_codon, args.seq_length)
    smiles += [convert_to_smiles(generated[i], char) for i in range(len(generated))]

#write smiles and calcualte properties of them    
print ('number of trial : ', len(smiles))
smiles = list(set([s.split('E')[0] for s in smiles]    ))
print ('number of generate smiles (after remove duplicated ones) : ', len(smiles))
ms = [Chem.MolFromSmiles(s) for s in smiles]
ms = [m for m in ms if m is not None]
print ('number of valid smiles : ', len(ms))
with open(args.result_filename, 'w') as w:
    w.write('smiles\n')
    for m in ms:
        w.write('%s\n' %(Chem.MolToSmiles(m)))
