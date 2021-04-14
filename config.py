import os
from dataclasses import dataclass


QUESTIONS_PATH = './data/questions_embeddings.npy'
DATA_PATH = './data/chatbot_data.txt'
DOT = '.'

@dataclass
class Config:
    Data: str=DATA_PATH if os.path.isfile(DATA_PATH) else DOT+DATA_PATH
    Questions: str=QUESTIONS_PATH if os.path.isfile(QUESTIONS_PATH) else DOT+QUESTIONS_PATH
    BERTMultiLingual: str='bert-base-multilingual-cased'