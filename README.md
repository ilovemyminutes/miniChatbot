# mini Chatbot
simple chat bot using pretrained BERT from huggingface


## Offline Task
```python
>>> python export_question_embeddings.py
```

## Run
```python
>>> python chatbot.py --sim-type cos
*********************************
환영합니다! 간단한 챗봇을 사용해보세요
*********************************
할 말을 입력해주세요(종료시 '대화종료' 입력): 내가 오늘 너를 만드느라 좀 힘들었어 
챗봇: 오늘은 힘내려 하지 말아요. 저에게 기대세요.
할 말을 입력해주세요(종료시 '대화종료' 입력): 대화종료 
*********************************
챗봇을 종료합니다. 다음에 또 만나요!
*********************************
```
- `sim_type`: 문장 간 유사도 측정에 활용할 측정 방법을 설정합니다. Cosine Similarity(`'cos'`), L1-Norm(`'l1'`), L2-Norm(`'l2'`)의 3가지 측정 방법을 지원합니다.(default. `'cos'`)

## Dataset
- [Chatbot_data](https://github.com/songys/Chatbot_data), [songys](https://github.com/songys)