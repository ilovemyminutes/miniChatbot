import os
from dataclasses import dataclass
import torch



@dataclass
class Config:
    ChatbotData: str = "./data/chatbot_data.txt"
    ParaKQCData: str = "./paraKQC/data/paraKQC_v1.txt"
    Questions: str = "./data/questions_embeddings.npy"
    BERTMultiLingual: str = "bert-base-multilingual-cased"
    Device: str = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')