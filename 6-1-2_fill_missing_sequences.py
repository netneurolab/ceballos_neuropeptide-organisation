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
files = glob.glob('./data/evo/03codalign/*.fa')
files.sort()

# create directory for filled files if it not already exists
if not os.path.exists('./data/evo/04codalign_filled'):
    os.makedirs('./data/evo/04codalign_filled')

for file in files:
    gene = os.path.basename(file).split('_')[0]
    print(gene)
    # per file read the protein sequences
    txt = np.loadtxt(file, dtype=str)

    if np.array_equal(txt, []):
        print(f'WARNING: Empty file for {gene}\n')
        continue

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

    # measure block length from the first to second header
    sequence_block = txt[1:spacing[0]].tolist()
    sequence_block = ''.join(sequence_block)
    sequence_length = len(sequence_block)

    # characters per line
    nchars = len(txt[1])
    empty_line = '-' * nchars

    # characters in last line
    nchars_last = len(txt[-1])
    empty_last_line = '-' * nchars_last
    
    # species are in the end of headers (last characters after _)
    ordered = []
    not_found = []
    for species in species_list:
        # find species line in the headers
        line_matches = np.char.rfind(txt, species) > 0
        header_idx = np.where(line_matches)[0]
        if np.array_equal(header_idx, []):
            empty_header = f'>XM_00000.0_XP_00000.0_{gene.lower()}_{species}'
            # add empty lines as many times as spacing
            sequence_block = [empty_line] * (spacing[0] - 2)
            sequence_block.insert(0, empty_header)
            sequence_block.append(empty_last_line)
            if len(''.join(sequence_block[1:])) != sequence_length:
                print(f'WARNING: Blank sequence length mismatch for {gene} {species}\n')
                break
            ordered.extend(sequence_block)
            not_found.append(species)
        else:
            sequence_block = txt[header_idx[0]:header_idx[0]+spacing[0]].tolist()
            ordered.extend(sequence_block)

    if not_found == []:
        print('All species found\n')
    else:
        print(f'Species not found: {", ".join(not_found)}\n')

    # save the ordered sequences
    with open(f'./data/evo/04codalign_filled/{gene}_codalign.fa', 'w') as f:
        for line in ordered:
            f.write(f'{line}\n')

