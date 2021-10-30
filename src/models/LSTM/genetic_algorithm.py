import numpy as np
import pandas as pd

class genetict_algorithm():
    def __init__(self, structure_gene: dict, n_chromosomes: int) -> None:
        genetic_matirial = []
        group_genetic = []
        generate_genetic_matirial = lambda x, y: [genetic_matirial.append(x) for i in range(y)]

        for i in structure_gene:
            _range, size, name = structure_gene[i]['range'], structure_gene[i]['size'], i
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

    def evolution(self, score):
        df = pd.DataFrame(self.population, columns=self.group_genetic)
        for i in df: df[i] = [j[i] for j in df[i]]
        df['score'] = score
        df = df.sort_values('score')

        print(df)
        size = self.n_chromosomes
        size_slice = int(len(df) * 0.20) 
        df = df.iloc[size_slice:, :]

        ax_df = None
        flag = True
        for i in range(size_slice):
            index = np.random.randint(0, (size-size_slice))
            if (flag): 
                flag = not flag
                ax_df = df.iloc[index:index+1, :]
            else: ax_df = ax_df.append(df.iloc[index:index+1, :])



        print(df)

structure_gene = {
    "dropout": {
        "range": (0.20, 0.50),
        "size": 3
    },
    "neurons": {
        "range": (10, 100),
        "size": 3
    },
    "activation": {
        "range": ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential'],
        "size": 3
    }
}

size = 20
score = [np.random.uniform(0, 1) for i in range(size)]

ga = genetict_algorithm(structure_gene, size)
population = ga.population
# print(population)
ga.evolution(score)