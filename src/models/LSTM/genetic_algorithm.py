import numpy as np
from numpy.core.defchararray import index
import pandas as pd
from pandas.core.base import DataError
from pandas.core.frame import DataFrame

class genetict_algorithm():
    def __init__(self, gene_structure: dict, n_chromosomes: int) -> None:
        self.gene_structure = gene_structure
        genetic_matirial = []
        group_genetic = []
        generate_genetic_matirial = lambda x, y: [genetic_matirial.append(x) for i in range(y)]

        for i in gene_structure:
            _range, size, name = gene_structure[i]['range'], gene_structure[i]['size'], i
            generate_genetic_matirial(_range, size)
            [group_genetic.append((f'{name}_{j}')) for j in range(size)]

        self.genetic_matirial = genetic_matirial
        self.group_genetic = group_genetic
        self.n_chromosomes = n_chromosomes

        self.generete_population()

    def generete_population(self):
        self.population = np.array([self.generete_chromosomes() for i in range(self.n_chromosomes)])

    def generete_chromosomes(self):
        values = []
        for i in self.genetic_matirial:
            _type = type(i[0])
            _low, _high = (0, len(i)) if _type == str else i
            str_type = 'int' if _type in [str, int] else 'float'

            oprtion = {
                'int': np.random.randint,
                'float': np.random.uniform,
            }

            n_rand = oprtion.get(str_type)(_low, _high)
            ax_value = i[n_rand] if _type == str else n_rand
            values.append(ax_value)

        return [{i: j} for i, j in zip(self.group_genetic ,values)]

    def evolution(self, score: list):
        df = pd.DataFrame(self.population, columns=self.group_genetic)
        for i in df: df[i] = [j[i] for j in df[i]]
        df['score'] = score
        df = df.sort_values('score')

        size = self.n_chromosomes
        percent_20 = int(len(df) * 0.20) 
        df = df.iloc[percent_20:, :]

        ax_df = None
        rand_list = np.arange(0, (size-percent_20))
        np.random.shuffle(rand_list)
        rand_list = rand_list[:4]
        for i in rand_list:
            if (type(ax_df) != DataFrame): ax_df = df.iloc[i:i+1, :]
            else: ax_df = ax_df.append(df.iloc[i:i+1, :])

        size_of_80p = (size - percent_20) 
        percent_50_of_80p = int(size_of_80p * 0.5)

        df_weak = df.iloc[:percent_50_of_80p, :].reset_index(drop='index')
        df_strong = df.iloc[percent_50_of_80p:, :].reset_index(drop='index')

        indexs = np.arange(percent_50_of_80p)
        np.random.shuffle(indexs)
        size_of_80p_of_25p = int(percent_50_of_80p * 0.25)
        cross_index_strong_w_weak = indexs[:size_of_80p_of_25p]

        for i in cross_index_strong_w_weak:
            a_chromosome, b_chromosome = df_weak[df_weak.index == i], df_strong[df_strong.index == i]
            df_weak, df_strong = df_weak.drop(index=i), df_strong.drop(index=i)
            a_chromosome, b_chromosome = self.cross_chromosome(a_chromosome, b_chromosome)
            ax_df = ax_df.append(a_chromosome)
            ax_df = ax_df.append(b_chromosome)

        ax_df = self.cross_df(df_weak, ax_df)
        df = self.cross_df(df_strong, ax_df)
        self.mutation(df)
    
    def mutation(self, df: DataFrame) -> np.array:
        for i in df.index:
            if (np.random.uniform(0, 1) > 0.90):
                size = len(df.columns) - 1
                n_genes = np.random.randint(0, int(size * 0.3)) + 1
                mutations_genes = np.arange(0, size)
                np.random.shuffle(mutations_genes)
                mutations_genes = mutations_genes[:n_genes]
                for j in mutations_genes:
                    feature = df.columns[j]
                    gene = self.gene_structure[feature.split('_')[0]]['range']

                    _type = type(gene[0])
                    _low, _high = (0, len(gene)) if _type == str else gene
                    str_type = 'int' if _type in [str, int] else 'float'

                    oprtion = {
                        'int': np.random.randint,
                        'float': np.random.uniform,
                    }

                    n_rand = oprtion.get(str_type)(_low, _high)
                    ax_value = gene[n_rand] if _type == str else n_rand
                    df.iloc[i:i+1, j:j+1] = ax_value

        self.score = df['score'].values
        for i in df:
            df[i] = [{i: j} for j in df[i]]
        
        self.population =  df.drop(columns='score').to_numpy()

    def cross_df(self, df: DataFrame, df_of_return: DataFrame) -> DataFrame:
        df = df.reset_index(drop='index')
        for i in np.arange(0, len(df), 2):
            a_chromosome, b_chromosome = df[df.index == i], df[df.index == i + 1]
            df = df.drop(index=[i])
            a_chromosome, b_chromosome = self.cross_chromosome(a_chromosome, b_chromosome)
            df_of_return = df_of_return.append(b_chromosome)
            df_of_return = df_of_return.append(a_chromosome)
        
        return df_of_return.reset_index(drop='index')

    def cross_chromosome(self, a_chromosome: DataFrame, b_chromosome: DataFrame) -> list:
        index = 0
        for i in self.gene_structure:
            size = self.gene_structure[i]['size']
            ax_index = index + size

            slice = np.random.randint(0, size)
            a_chr = a_chromosome.iloc[:, index:ax_index].values
            b_chr = b_chromosome.iloc[:, index:ax_index].values

            ab_chr = np.append(a_chr[0][:slice], b_chr[0][slice:])
            ba_chr = np.append(b_chr[0][:slice], a_chr[0][slice:])

            a_chromosome.iloc[:, index:ax_index] = ab_chr
            b_chromosome.iloc[:, index:ax_index] = ba_chr

            index = ax_index

        return [a_chromosome, b_chromosome]