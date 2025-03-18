import os
import jieba
from zhipuai import ZhipuAI
from bm25 import BM25

'''
基于RAG来解答人工智能客服问题
用bm25做召回
使用zhipu的api作为大模型
'''


# 智谱的api作为我们的大模型
def call_large_model(prompt):
    client = ZhipuAI(api_key="")  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4-flashx",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    response_text = response.choices[0].message.content
    return response_text


class SimpleRAG:
    def __init__(self, folder_path="人工智能客服"):
        self.load_data(folder_path)

    def load_data(self, folder_path):
        self.ai_data = {}
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                    intro = file.read()
                    hero = file_name.split(".")[0]
                    self.ai_data[hero] = intro
        corpus = {}
        self.index_to_name = {}
        index = 0
        for hero, intro in self.ai_data.items():
            corpus[hero] = jieba.lcut(intro)
            self.index_to_name[index] = hero
            index += 1
        self.bm25_model = BM25(corpus)
        return

    def retrive(self, user_query):
        scores = self.bm25_model.get_scores(jieba.lcut(user_query))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        hero = sorted_scores[0][0]
        text = self.ai_data[hero]
        return text

    def query(self, user_query):
        print("user_query:", user_query)
        print("=======================")
        retrive_text = self.retrive(user_query)
        # print("retrive_text:", retrive_text)
        # print("=======================")
        prompt = f"请根据以下从数据库中获得的实际解决方案，回答用户问题：\n\n实际解决方案：\n{retrive_text}\n\n用户问题：{user_query}"
        response_text = call_large_model(prompt)
        print("模型回答：", response_text)
        print("=======================")


if __name__ == "__main__":
    # 记录运行时间
    import time
    start_time = time.time()
    rag = SimpleRAG()
    user_query = "信号不好怎么办"
    rag.query(user_query)
    end_time = time.time()
    print("运行时间：", end_time - start_time)