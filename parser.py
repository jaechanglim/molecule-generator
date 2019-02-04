with open('chembl_data') as f:
    lines = f.read().split('\n')[:-1]
    lines = [l.split() for l in lines]
    lines = [l for l in lines if l[-1]!='None'] 
    smiles = list(set([l[3] for l in lines if l[1]=='egfr' and float(l[-1])>6]))
with open('egfr_smiles.txt', 'w') as w:
    for s in smiles:
        w.write(s+'\n')
            
