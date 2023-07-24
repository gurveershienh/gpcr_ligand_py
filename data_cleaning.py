##importing necessary libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from configuration import names, coded_gpcrs, MOLWT_LOWER, MOLWT_UPPER, LOGP_LOWER, LOGP_UPPER

## loading data into Dataframe objects
ligand_df  = pd.read_csv('gpcr_ligands.tsv', sep='\t').drop_duplicates()
interactions_df = pd.read_csv('interactions_active.tsv', sep='\t').drop_duplicates()
targets_df = pd.read_csv('targets.tsv', sep='\t').drop_duplicates()
coded_gpcr_df = pd.read_csv('coded_gpcr_list.csv', index_col=0)

##normalizing column names
ligand_df.columns=ligand_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
interactions_df.columns=interactions_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
targets_df.columns=targets_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

##formatting  gpcr_name values
for name, code in zip(names, coded_gpcrs):
    targets_df = targets_df.replace(name, code)

targets_df['first_seg'] = [name.split(' ', maxsplit=1)[0] for name in targets_df.gpcr_name]


##typecasting on select columns
vals = pd.to_numeric(interactions_df['value'], errors='coerce')
interactions_df['value'] = vals
ligand_df = ligand_df.astype({'ligand_name': 'string'})
interactions_df = interactions_df.astype({'database_ligand_id': 'string'})


##dropping unnecessary attributes (increases efficiency)
ligand_df.drop(['cid','molecular_formula', 'inchi_std._id', 'inchi_key', 'iupac_name', 'isomeric_smiles'], axis=1, inplace=True)
interactions_df.drop(['inchi_key', 'parameter', 'unit', 'database_source', 'database_target_id', 'reference'], axis=1, inplace=True)
targets_df.drop(['gpcr_name','gene_name', 'species', 'fasta_sequence'], axis=1, inplace=True)
coded_gpcr_df.drop(['gpcr_name','second_seg'], axis=1,inplace=True)

##filtering ligand_df by xlogP and MW thresholds
ligand_df.drop(ligand_df[(ligand_df.xlogp > LOGP_UPPER) | (ligand_df.xlogp < LOGP_LOWER) | (ligand_df.molecular_weight > MOLWT_UPPER) | (ligand_df.molecular_weight < MOLWT_LOWER)].index, inplace=True)

##removing null vals + duplicates and renaming database_ligand_id
interactions_df.dropna(subset=['database_ligand_id', 'uniprot_id', 'value'], inplace=True)
interactions_df.rename(columns={'database_ligand_id': 'ligand_name'}, inplace=True)
coded_gpcr_df.drop_duplicates(inplace=True)


##print dataframes
print(ligand_df)
print(targets_df)
print(interactions_df)
print(len(np.unique(np.array(interactions_df['ligand_name']))))
print(coded_gpcr_df)

##aggregate and join to find max interaction value for each ligand
max_affinity_df = interactions_df.groupby(['ligand_name'], as_index=False)['value'].max()
max_affinity_df = interactions_df.merge(max_affinity_df, how='inner', on=['ligand_name', 'value'])
max_affinity_df.drop_duplicates(subset=['ligand_name'],inplace=True)
max_affinity_df.dropna(inplace=True)

#check agg/join result
print(max_affinity_df)

##outer joining original dataframes
gpcr_targets_df = targets_df.merge(interactions_df, how='inner', on=['uniprot_id'])
gpcr_ligand_df = gpcr_targets_df.merge(ligand_df, how='inner', on=['ligand_name'])
gpcr_max_aff_df = gpcr_targets_df.merge(max_affinity_df, how='inner', on=['uniprot_id', 'value', 'ligand_name'])
encoded_df = gpcr_ligand_df.merge(coded_gpcr_df, how='inner', on=['first_seg'])
print(gpcr_targets_df)
print(gpcr_ligand_df)
print(gpcr_max_aff_df)
##drop duplicates/null values
encoded_df.drop_duplicates(subset=['ligand_name'],inplace=True)
encoded_df.dropna(subset=['canonical_smiles', 'ligand_name', 'uniprot_id', 'gpcr_binding_encoded'],inplace=True)

#check results
print(encoded_df)
print(len(np.unique(np.array(encoded_df['ligand_name']))))
encoded_df.to_csv('basic_descriptors.csv', index=False)




