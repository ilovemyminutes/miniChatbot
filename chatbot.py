import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from config import Config
from similarity import get_cosine_similarity, get_l1_distance, get_l2_distance

EXIT = "대화종료"


class VanillaChatbot:
    def __init__(self, sim_type: str='cos'):
        self.model = AutoModel.from_pretrained(Config.BERTMultiLingual)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.BERTMultiLingual)
        self.questions = self.load_questions(Config.Questions)
        self.answers = pd.read_csv(Config.Data)["A"].tolist()
        self.measure = self.get_similarity_measure(sim_type=sim_type) # 유사도 측정 객체
        return

    def query(self, question: str, return_answer=False):
        question_tokenized = self.tokenizer(question, return_tensors="pt")
        question_embedded = self.model(**question_tokenized).pooler_output
        similar_question_id = self.get_similar_question_id(question_embedded)
        answer = self.answers[similar_question_id]
        print('챗봇:', answer)
        if return_answer:
            return answer

    def get_similar_question_id(self, q_emb):
        similarities = self.measure(self.questions, q_emb)
        similar_question_id = torch.argmax(similarities).item()
        return similar_question_id

    @staticmethod
    def load_questions(root: str = Config.Questions):
        questions = torch.tensor(np.load(root))
        return questions
    
    @staticmethod
    def get_similarity_measure(sim_type: str):
        if sim_type == 'cos':
            measure = get_cosine_similarity
        elif sim_type == 'l1':
            measure = get_l1_distance
        elif sim_type == 'l2':
            measure = get_l2_distance
        return measure



if __name__ == "__main__":
    print('*'*150)
    print('환영합니다! 간단한 챗봇을 사용해보세요')
    print('*'*150)
    bot = VanillaChatbot()
    while True:
        text = input("할 말을 입력해주세요(종료시 '대화종료' 입력): ")
        if text == EXIT:
            break
        bot.query(text)
    print('*'*150)
    print('챗봇을 종료합니다. 다음에 또 만나요!')
    print('*'*150)
