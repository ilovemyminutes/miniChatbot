import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from config import Config

EXIT = '대화종료'

class VanillaChatbot:
    def __init__(self):
        self.model = AutoModel.from_pretrained(Config.BERTMultiLingual)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.BERTMultiLingual)
        
        self.questions = self.load_questions(Config.Questions)
        self.answers = pd.read_csv(Config.Data)['A'].tolist()

    def query(self, question: str, return_answer=False):
        q_tok = self.tokenizer(question, return_tensors='pt')
        q_emb = self.model(**q_tok).pooler_output
        similar_q_id = self.get_similar_question_id(q_emb)
        answer = self.answers[similar_q_id]
        print(answer)
        if return_answer:
            return answer
    
    def get_similar_question_id(self, q_emb):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        similarities = cos(self.questions, q_emb)
        return torch.argmax(similarities).item()


    @staticmethod
    def load_questions(root: str=Config.Questions):
        questions = torch.tensor(np.load(root))
        return questions


if __name__ == '__main__':
    bot = VanillaChatbot()
    while True:
        text = input("할 말을 입력해주세요(종료시 '대화종료' 입력): ")
        if text == EXIT:
            break
        bot.query(text)

    




