import time

import dotenv
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
from konlpy.tag import Okt
import numpy as np
import faiss
import os
import pickle

import getArticle

# OpenAI API 키 설정
dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)
api_key = os.environ["OPENAI_API_CD"]


class QuestionArticleMatcher:
    def __init__(self, model_name='distiluse-base-multilingual-cased-v1', index_path='sentence_index.faiss',
                 data_path='qa_pairs.pkl'):
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.data_path = data_path
        self.questions = []
        self.articles = []
        self.load_data()

    def load_data(self):
        # FAISS 인덱스 로드
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"Loaded existing index from {self.index_path}")
        else:
            self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
            print("Created new index")

        # 질문과 기사 데이터 로드
        if os.path.exists(self.data_path):
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
                self.questions = data['questions']
                self.articles = data['articles']
            print(f"Loaded existing data from {self.data_path}")
        else:
            print("No existing data found")

    def save_data(self):
        # FAISS 인덱스 저장
        faiss.write_index(self.index, self.index_path)

        # 질문과 기사 데이터 저장
        with open(self.data_path, 'wb') as f:
            pickle.dump({'questions': self.questions, 'articles': self.articles}, f)

        print(f"Saved index to {self.index_path} and data to {self.data_path}")

    def add_question_article_pairs(self, questions, articles):
        assert len(questions) == len(articles), "질문과 기사의 수가 일치해야 합니다."
        self.questions.extend(questions)
        self.articles.extend(articles)

        new_embeddings = self.model.encode(questions)
        self.index.add(new_embeddings)

        self.save_data()

    def find_most_similar_article(self, new_question):
        new_question_embedding = self.model.encode([new_question])[0]

        D, I = self.index.search(np.array([new_question_embedding]), 1)
        most_similar_index = I[0][0]
        similarity_score = 1 - (D[0][0] / 2)

        return {
            "original_question": self.questions[most_similar_index],
            "article": self.articles[most_similar_index],
            "similarity_score": similarity_score
        }


class QuestionAnsweringSystem:
    def __init__(self):
        self.okt = Okt()
        self.matcher = QuestionArticleMatcher()
        self.llm = ChatOpenAI(temperature=0.7, api_key=api_key)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["chat_history", "human_input"],
                template="You are a helpful AI assistant. Chat History: {chat_history}\nHuman: {human_input}\nAI: "
            ),
            memory=self.memory
        )
        self.conversation_with_article = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["chat_history", "input"],
                template="You are a helpful AI assistant. Please summarize article. Chat History: {chat_history}\n{"
                         "input}\nAI:"
            ),
            memory=self.memory
        )

    def process_sentence(self, sentence: str) -> str:
        nouns = self.okt.nouns(sentence)
        noun_string = ' '.join(nouns)
        return noun_string
    def generate_keyword(self, user_keyword):
        keyword_gen_template = """아래의 질문을 몇가지 키워드로 요약해줘.
        질문: {question}"""
        prompt = PromptTemplate.from_template(keyword_gen_template)
        # prompt.format(question="아이유 몇살이야?")

        chat_model = ChatOpenAI(openai_api_key=api_key)
        keyword = chat_model.predict(prompt.format(question=user_keyword))
        return keyword

    def add_question_article_pairs(self, questions, articles):
        self.matcher.add_question_article_pairs(questions, articles)

    def answer_question(self, question):
        start_process = time.time()
        # print(f"1. 시작: {start}")
        result = self.matcher.find_most_similar_article(question)
        find_similarity = time.time()
        print(f"1. 유사 기사 검색 : {(find_similarity - start_process):.2f}")
        if result["similarity_score"] >= 0.5:
            print(
                f"Based on a similar question: '{result['original_question']}', here's the relevant article:\n\n{result['article']}")
            combined_input = f"Human Input: {question}\nArticle: {result['article']}"
            print("* 유사 기사 검색 완료.")

            start_get_answer = time.time()
            result_answer = self.conversation_with_article.predict(input=combined_input)
            end_get_answer = time.time()
            print(f"2-1. gpt 답변 완료: {(end_get_answer - start_get_answer):.2f}")

            end_process = time.time()
            print(f"**** 프로세스 총 누적 소요 시간: {(end_process - start_process):.2f}")
            return result_answer
        else:
            print("* 유사 기사 검색 실패. 신규 기사 수집")
            start_generate_keyword = time.time()
            # keyword = self.generate_keyword(question)
            keyword = self.process_sentence(question)
            end_generate_keyword = time.time()
            print(f"2-1. 키워드 추출: {(end_generate_keyword - start_generate_keyword):.2f}")

            start_find_articles = time.time()
            articles = getArticle.getArticleDetailBulkWithStr(keyword)[0:2000]
            end_find_articles = time.time()
            print(f"2-2. 기사 수집: {(end_find_articles - start_find_articles):.2f}")
            # print("!!!!!!", articles)
            combined_input = f"Human Input: {question}\nArticle: {articles}"

            start_get_response = time.time()
            response = self.conversation_with_article.predict(input=combined_input)
            end_get_response = time.time()
            print(f"2-3. gpt 답변 완료: {(end_get_response - start_get_response):.2f}")
            # response = self.conversation.predict(human_input=question)
            qa_system.add_question_article_pairs([question], [response])

            end_process = time.time()
            print(f"**** 프로세스 총 누적 소요 시간: {(end_process - start_process):.2f}")
            return response


# 사용 예시
qa_system = QuestionAnsweringSystem()

if __name__ == "__main__":
    # 기존 데이터가 없을 경우에만 새로운 질문-기사 쌍 추가
    if len(qa_system.matcher.questions) == 0:
        existing_questions = [
            "인공지능의 미래는 어떻게 될까요?",
            "기후 변화가 생태계에 미치는 영향은 무엇인가요?",
            "암 치료의 최신 기술은 무엇인가요?",
            "태양계에서 가장 큰 행성은 무엇인가요?",
            "프랑스 대혁명의 주요 원인 3가지를 설명해주세요.",
            "매일 물을 8잔 마셔야 한다는 말은 사실인가요?",
            "인지부조화란 무엇이며, 일상생활의 예를 들어주세요."
        ]
        existing_articles = [
            "인공지능 기술의 발전으로 인해 미래에는 많은 일자리가 자동화될 것으로 예상됩니다. 그러나 동시에 새로운 형태의 일자리도 생겨날 것입니다...",
            "기후 변화로 인한 생태계 변화는 다음과 같습니다: 1) 생물 다양성 감소, 2) 해수면 상승으로 인한 해안 생태계 파괴, 3) 극단적인 기후 현상 증가...",
            "최근 암 치료 기술 중 가장 주목받는 것은 면역 치료입니다. 이 기술은 환자의 면역 체계를 강화하여 암 세포를 공격하도록 합니다...",
            "태양계에서 가장 큰 행성은 목성입니다. 목성은 지구 질량의 약 318배, 부피는 약 1,321배에 달하는 거대한 가스 행성입니다.",
            "프랑스 대혁명의 주요 원인 3가지는 다음과 같습니다: 1.절대왕정의 재정 위기, 2.구제도(앙시앵 레짐)의 불평등한 사회 구조, 3.계몽사상의 영향과 자유, 평등 사상의 확산",
            "매일 물 8잔 이라는 규칙은 일반적인 지침일 뿐, 과학적 근거는 부족합니다.실제 필요한 수분 섭취량은 개인의 체격, 활동량, 기후 등에 따라 다릅니다.갈증을 느낄 때 물을 마시고, 소변 색이 연한 노란색을 유지하도록 하는 것이 좋은 방법입니다.",
            "인지부조화는 개인의 신념, 태도, 행동 사이에 모순이 생길 때 발생하는 심리적 불편함을 말합니다. 예를 들어, 흡연이 건강에 해롭다는 것을 알면서도 계속 담배를 피우는 경우, 또는 환경 보호의 중요성을 강조하면서 일회용품을 자주 사용하는 경우 등이 인지부조화의 일상적 예시입니다."
        ]
        qa_system.add_question_article_pairs(existing_questions, existing_articles)

    # 질문 테스트
    questions = [
        "AI가 우리의 삶을 어떻게 바꿀까요?",
        "지구 온난화의 주요 원인은 무엇인가요?",
        "암 치료 기술로 무엇이 있나요?",
        "태양계에서 제일 큰 행성이 뭐에요?",
        "프랑스 대혁명의 원인을 알고싶어.",
        "물을 8잔씩 마셔야 해?",
        "인지부조화 설명 및 예시",
        #
        # "오물풍선 피해사례",
        # "12월 3일 계엄령 사태 설명",
        # "전청조 사기액수",
        # "12월 3일 계엄사령관 설명",
        # "아이유 근황 알고싶어",
        # "이재명 활동",
        # "AWS 관련 소식",
    ]

    for question in questions:
        print(f"질문: {question}")
        answer = qa_system.answer_question(question)
        print(f"답변: {answer}\n")
