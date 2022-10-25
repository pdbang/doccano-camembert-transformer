import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from transformers import CamembertTokenizerFast

import scripts.utils as utils

class JsonlDoccano:

    def __init__(self, filepath, camembert=True):
        if camembert:
            self.TOKENIZER = CamembertTokenizerFast.from_pretrained(
                'camembert-base',
                do_lower_case=True, encoding='utf-8').tokenize
        else:
            self.TOKENIZER = str.split
        
        self.camembert = camembert
            
        self.filepath = filepath
        self.json_file = utils.read_jsonl(filepath=self.filepath, encoding='utf-8').to_pandas(self.TOKENIZER)
        self.df = self.to_pandas()
        self.labels = self.df[self.df['label'] != 'O']
        self.sentences = self.labels.id_phrase.unique()


    def to_pandas(self):
        list_df = []
        for entry in self.json_file:
            list_df.append(list(entry))
        df = pd.DataFrame(list_df, columns=['token', 'label', 'id_phrase'])
        return df

    def run_export(self, resultpath, augmented=False):
        if augmented:
            self.df_augmented.to_parquet(resultpath)
        else:
            self.df.to_parquet(resultpath)
        return