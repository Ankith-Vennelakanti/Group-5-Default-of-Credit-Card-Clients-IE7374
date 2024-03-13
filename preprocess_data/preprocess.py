import pandas as pd
# class pre_process:
    # def __init__(self) -> None:
    #     # import pandas as pd
    #     pass
    
def process(self, path):
    df = pd.read_excel(io = path, index_col=False)
    df = df.drop(columns=['ID','EDUCATION', 'MARRIAGE','AGE'])

    return df
    
def remove_target(self, df):
    df = df.drop(columns=['default payment next month'])

    return df

    