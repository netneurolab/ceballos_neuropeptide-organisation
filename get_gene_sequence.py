# %%
import requests
import time
import pandas as pd
import numpy as np
import itertools

species_list = [('Homo sapiens', 'hsapiens'),
                ('Pan troglodytes',  'ptroglodytes'),
                ('Macaca mulatta',  'mmulatta'),
                ('Mus musculus',  'mmusculus'),
                ('Bos taurus',  'btaurus'),
                ('Dasypus novemcinctus',  'dnovemcinctus'),
                ('Sarcophilus harrisii',  'sharrisii'),
                ('Ornithorhynchus anatinus',  'oanatinus'),
                ('Gallus gallus',  'ggallus'),
                ('Xenopus tropicalis',  'xtropicalis'),
                ('Danio rerio',  'drerio'),
                ('Carcharodon carcharias',  'ccarcharias'),
                ('Petromyzon marinus',  'pmarinus')]


# loop through receptor list
# load peptide genes
peptide_genes = pd.read_csv('data/peptide_genes_ahba_Schaefer400.csv', index_col=0)
genes = peptide_genes.columns.to_list()

# load gene list
receptor_list = pd.read_csv('data/receptor_list.csv')['Gene'].to_list()
gene_list = pd.read_csv('data/gene_list.csv')['Gene'].to_list()
receptor_list = gene_list

# %%

# create dataframe to keep track of when gene did not yield a result. size is receptor by species
df = pd.DataFrame(np.full((len(receptor_list), len(species_list)), np.nan, dtype=object), 
                  index=receptor_list, columns=[species[1] for species in species_list], dtype=object)

complete_nucseq = ""
complete_proseq = ""

# loop through genes
for receptor in receptor_list:

    if receptor == 'CORT':
        continue
    
    for (species, species_id) in species_list:
        
        # get id
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=nuccore&term={species}[Organism] AND {receptor}"
        search_results = requests.get(url).text
        time.sleep(1)

        # find all occurences of <Id> and get text between <Id> and </Id>
        # assert if <Id>  is not found in search_results, check for ValueError
        ids = []
        start = 0
        while True:
            try:
                start = search_results.index('<Id>', start)
            except ValueError:
                break
            end = search_results.index('</Id>', start)
            gid = search_results[start+4:end]
            ids.append(gid)
            start = end
        
        # proceed to fetch data
        fetch_data = True
        
        # check if ids is empty
        if ids == []:
            print(f"   WARNING: Gene {receptor} did not return any results for {species}")
            df.loc[receptor, species_id] = False
            fetch_data = False
        
        # try:
        #     start = search_results.index('<Id>')
        # except ValueError:
        #     print(f"Gene {gene} not found")
        #     continue
        # end = search_results.index('</Id>')
        # gid = search_results[start+4:end]
        if fetch_data:
            # check which id is correct
            # loop through ids until correct gene is found
            for gid in ids:
                url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=nuccore&id={gid}"
                esummary = requests.get(url).text
                time.sleep(1)
                if receptor in esummary.upper():
                    break

            # get title by indexing <Item Name="Title" Type="String"> and </Item>
            start = esummary.index('<Item Name="Title" Type="String">')
            end = esummary.index('</Item>', start)
            title = esummary[start+33:end]

            print(title)
            df.loc[receptor, species_id] = True
            
            # fetch nucleotide sequence
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={gid}&rettype=fasta&retmode=text"
            nucseq = requests.get(url).text
            time.sleep(1)
            cds_accession = nucseq.split(' ')[0][1:]

            # search for corresponding protein id
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=nuccore&db=protein&id={gid}"
            elink = requests.get(url).text
            time.sleep(1)
            # index start of protein list
            try:
                start = elink.index('<LinkName>nuccore_protein</LinkName>')
            except ValueError:
                print(f"   WARNING: Gene {receptor} did not return a corresponding protein in {species}")
                df.loc[receptor, species_id] = False
                continue
            # index first id
            start = elink.index('<Id>', start)
            end = elink.index('</Id>', start)
            pid = elink[start+4:end]

            # fetch protein sequence
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id={pid}&rettype=fasta&retmode=text"
            proseq = requests.get(url).text
            time.sleep(1)
            aas_accession = proseq.split(' ')[0][1:]


            # replace fasta header with >{cds_accession}_{aas_accession}_{receptor}_{species_id}
            nucseq = nucseq.split('\n')[1:]
            nucseq = '\n'.join(nucseq)
            complete_nucseq = f"{complete_nucseq}" + \
                            f">{cds_accession}_{aas_accession}_{receptor.lower()}_{species_id}\n{nucseq}\n"

            proseq = proseq.split('\n')[1:]
            proseq = '\n'.join(proseq)
            complete_proseq = f"{complete_proseq}" + \
                            f">{cds_accession}_{aas_accession}_{receptor.lower()}_{species_id}\n{proseq}\n"


        # save as formatted fasta file
        if species == 'Petromyzon marinus':
            with open(f'./results/fasta/{receptor}_nucseq.fasta', 'w') as f:
                f.write(complete_nucseq)
            
            with open(f'./results/fasta/{receptor}_proseq.fasta', 'w') as f:
                f.write(complete_proseq)

            complete_nucseq = ""
            complete_proseq = ""
            df.to_csv('./results/fasta/found_genes.csv')

    # # get gene sequence
    # url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={gid}&rettype=fasta&retmode=text"
    # html_content = requests.get(url).text
    # gene_sequence = html_content

    # # save gene sequence to file
    # with open(f'./data/{gene}.fasta', 'w') as f:
    #     f.write(gene_sequence)


# %%

for species, receptor in itertools.product(species_list, receptor_list[:1]):
    print(receptor)
    print(species)
    # get id
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=protein&term=({species}[Organism] AND {receptor}*"
    search_results = requests.get(url).text
    time.sleep(1)

    # find all occurences of <Id> and get text between <Id> and </Id>
    # assert if <Id>  is not found in search_results, check for ValueError
    ids = []
    start = 0
    try:
        start = search_results.index('<Id>', start)
    except ValueError:
        print(f"   WARNING: Gene {receptor} did not return any results for {species}")
        continue
    end = search_results.index('</Id>', start)
    gid = search_results[start+4:end]
    start = end
    
    
    # try:
    #     start = search_results.index('<Id>')
    # except ValueError:
    #     print(f"Gene {gene} not found")
    #     continue
    # end = search_results.index('</Id>')
    # gid = search_results[start+4:end]
        
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=protein&id={gid}"
    esummary = requests.get(url).text    
    time.sleep(1)
    
    # get title by indexing <Item Name="Title" Type="String"> and </Item>
    start = esummary.index('<Item Name="Title" Type="String">')
    end = esummary.index('</Item>', start)
    title = esummary[start+33:end]
    
    # get reference by indexing <Item Name="Extra" Type="String"> and </Item>
    start = esummary.index('<Item Name="Extra" Type="String">')
    # index |ref| and next | to get reference
    start = esummary.index('|ref|', start)
    end = esummary.index('|', start+1)
    reference = esummary[start+5:end]
    
    print(title)
# %%
