import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from collections import Counter

# From proteinclass
from proteinclass.dataloader import read_data

def plot_seq_count(ax, df, data_name):
    sns.histplot(ax=ax, data=df, x='Sequence length', kde=True)
    ax.set_title(f'{data_name}')
    ax.grid(True)

def get_code_freq(df, data_name):

    df = df.apply(lambda x: " ".join(x))

    codes = []
    for i in df: # concatination of all codes
        codes.extend(i)

    codes_dict= Counter(codes)
    codes_dict.pop(' ') # removing white space

    print(f'Codes: {data_name}')
    print(f'Total unique codes: {len(codes_dict.keys())}')

    df = pd.DataFrame({'Code': list(codes_dict.keys()), 'Freq': list(codes_dict.values())})
    return df.sort_values('Freq', ascending=False).reset_index()[['Code', 'Freq']]

def plot_code_freq(ax, df, data_name):
    ax.set_title(f'{data_name}')
    sns.barplot(ax=ax, x='Code', y='Freq', data=df)

def print_info(df, dset_name, latex=False):
    """ Print info on a dataframe. """
    print('-' * 50)
    print(f'Dataset partition: {dset_name}')
    print(f'Number of sequences: {len(df):d}')
    print(f'Number of unique family_accesion in {dset_name}: ', len(np.unique(df['family_accession'].values)))
    print(f'Number of unique family_id in {dset_name}: ', len(np.unique(df['family_id'].values)))
    print(f'Number of unique sequence_name in {dset_name}: ', len(np.unique(df['sequence_name'].values)))

    # Repartition of families in a dataset
    print('\nFamily distribution:')
    print(df.family_accession.value_counts())
    if latex:
        print('Head of table (LaTeX format):')
        print(df.head().to_latex(index=False))

    print('-' * 50 + '\n')

def getTransitionMatForSequence(transitions, default):
    """ Compute transition matrix for a given sequence. """
    df = pd.DataFrame(transitions)
    # We shift the sequence (first column i.e. df[0]) by one to the right
    df['shift'] = df[0].shift(-1)
    df['count'] = 1
    trans_mat = df.groupby([0, 'shift']).count().unstack().fillna(0)
    trans_mat.columns = trans_mat.columns.droplevel()
    return (default + trans_mat).fillna(0)

def makeHeatmap(df, familyAcc):
    """ Create Heatmap for a given family in the df DataFrame.
    The matrix is such that for each row, the column represents
    probability that the next letter is the one of the column. """
    mask = df.family_accession == familyAcc
    famSeqs = df.loc[mask, 'sequence'].reset_index(drop=True)
    AAs = [aa for aa in 'GALMFWKQESPVICYHRNDTXUBOZ']
    seqs = famSeqs.apply(lambda seq: [aa for aa in seq])
    default = pd.DataFrame([[0]*len(AAs)]*len(AAs), columns=AAs, index=AAs, dtype='int64')
    transMat = default.copy()
    for seq in seqs: #tqdm(seqs):
        transMat += getTransitionMatForSequence(seq, default)
    maskX = transMat.sum(axis=1) != 0
    maskY = transMat.sum(axis=0) != 0
    transMat = transMat.loc[maskX, maskY]
    transMat = transMat.div(transMat.sum(axis=1), axis=0)
    return transMat

def plot_heatmap(df, figname, familyAcc):
    """ Plot Heatmap for a given familyAcc in the df DataFrame. """
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(makeHeatmap(df, familyAcc), ax=ax)
    fig.savefig(figname, format='pdf', bbox_inches='tight')

def plot_sequence_len(df_train, df_val, df_test, figname, familyAcc=None):
    # Filter the datasets to only one family
    if familyAcc != None:
        df_train = df_train.loc[df_train.family_accession == familyAcc, :]
        df_val = df_val.loc[df_val.family_accession == familyAcc, :]
        df_test = df_test.loc[df_test.family_accession == familyAcc, :]

    # Plotting the sequence length distribution
    df_train['Sequence length'] = df_train['sequence'].apply(lambda x: len(x))
    df_val['Sequence length'] = df_val['sequence'].apply(lambda x: len(x))
    df_test['Sequence length'] = df_test['sequence'].apply(lambda x: len(x))

    fig, axes = plt.subplots(figsize=(15, 5), ncols=3)
    plot_seq_count(axes[0], df_train, 'Train')
    plot_seq_count(axes[1], df_val, 'Valid')
    plot_seq_count(axes[2], df_test, 'Test')
    fig.savefig(figname, format='pdf', bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    # Figures directory
    fig_dir = Path('figures')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Location of data folder
    data_path = '../input/random_split'
    print('Available data', os.listdir(data_path))

    # reading all data_partitions and visualizing (head) of the data
    df_train = read_data(data_path, 'train')
    df_val = read_data(data_path, 'dev')
    df_test = read_data(data_path, 'test')

    print_info(df_train, 'Train', latex=True)
    print_info(df_val, 'Valid')
    print_info(df_test, 'Test')

    # Plotting the labels distribution
    valCounts = pd.concat([pd.DataFrame(df_train.family_accession.value_counts()[:20]),
            pd.DataFrame(df_val.family_accession.value_counts()[:20]),
            pd.DataFrame(df_test.family_accession.value_counts()[:20])],
            axis=1)
    valCounts.columns = ['Train samples', 'Valid samples', 'Test samples']

    fig, ax = plt.subplots(figsize=(10, 7))
    valCounts.plot.bar(ax=ax, fontsize = 15, stacked=True)
    fig.savefig(fig_dir / 'labels_dist.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)

    # Plotting the sequence length distribution for the whole dataset
    # and for specific families (PF13649 is the most common one accross the dataset)
    plot_sequence_len(df_train, df_val, df_test, fig_dir / 'length_seq.pdf')
    plot_sequence_len(df_train, df_val, df_test, fig_dir / 'length_seq_13649.pdf', familyAcc='PF13649.6')
    plot_sequence_len(df_train, df_val, df_test, fig_dir / 'length_seq_00677.pdf', familyAcc='PF00677.17')

    # Plotting heatmap for some families in training datasets
    plot_heatmap(df_train, fig_dir / 'seq_heatmap_13649.pdf', familyAcc='PF13649.6')
    plot_heatmap(df_train, fig_dir / 'seq_heatmap_00677.pdf', familyAcc='PF00677.17')

    # Plotting the amino acid codes by computing the amino
    # acids distribution across each dataset
    train_code_freq = get_code_freq(df_train['sequence'], 'Train')
    val_code_freq = get_code_freq(df_val['sequence'], 'Val')
    test_code_freq = get_code_freq(df_test['sequence'], 'Test')

    fig, axes = plt.subplots(figsize=(14, 5), ncols=3)
    plot_code_freq(axes[0], train_code_freq, 'Train')
    plot_code_freq(axes[1], val_code_freq, 'Val')
    plot_code_freq(axes[2], test_code_freq, 'Test')
    fig.savefig(fig_dir / 'amino_acids_dist.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)
