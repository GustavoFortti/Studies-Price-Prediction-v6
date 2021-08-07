#data
#pred
#test
#train

class Report():
    def __init__(self) -> None:
        pass

    def set_df_origin(self, x, y):
        x['target'] = y
        self.df_org = x

    def set_df_result(self, df):
        self.df_res = df

    def validation_data(): # função de validação dos dados
        pass

    def compare_train(): # função de comparação do treinamento com outros trinos
        pass

    def calc_test(): # função sobre todos os calculos de testes
        pass

    def pred():
        pass