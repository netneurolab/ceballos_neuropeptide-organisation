# %%
import glob
import os
import numpy as np


species_list = ['hsapiens',
                'ptroglodytes',
                'mmulatta',
                'mmusculus',
                'btaurus',
                'dnovemcinctus',
                'sharrisii',
                'oanatinus',
                'ggallus',
                'xtropicalis',
                'drerio',
                'ccarcharias',
                'pmarinus']

# find all the files in the directory
proseq_files = glob.glob('./data/evo/01proseqs_align/*.fasta')
proseq_files.sort()

# create directory for reordered files if it not already exists
if not os.path.exists('./data/evo/02proseqs_ordered'):
    os.makedirs('./data/evo/02proseqs_ordered')

for proseq in proseq_files:
    gene = os.path.basename(proseq).split('_')[0]
    print(gene)
    # per file read the protein sequences
    txt = np.loadtxt(proseq, dtype=str)

    # find headers
    # header start with >
    headers = np.where(np.char.startswith(txt, '>'))[0]

    if len(headers) < 8:
        print(f'WARNING: Something went wrong with {gene}\n')
        continue

    # make sure space between headers is same
    spacing = np.diff(headers)
    spacing = np.unique(spacing)
    if len(spacing) > 1:
        print(f'WARNING: Unequal spacing between sequences for {gene}\n')
        continue
    
    # species are in the end of headers (last characters after _)
    ordered = []
    not_found = []
    for species in species_list:
        # find species line in the headers
        line_matches = np.char.rfind(txt, species) > 0
        header_idx = np.where(line_matches)[0]
        if np.array_equal(header_idx, []):
            not_found.append(species)
            continue
        sequence_block = txt[header_idx[0]:header_idx[0]+spacing[0]]
        ordered.extend(sequence_block.tolist())

    if not_found == []:
        print('All species found\n')
    else:
        print(f'The following species were not found for {gene}: {", ".join(not_found)}\n')

    # save the ordered sequences into new file
    with open(f'./data/evo/02proseqs_ordered/{gene}_proseq_ordered.fasta', 'w') as f:
        for line in ordered:
            f.write(f'{line}\n')

