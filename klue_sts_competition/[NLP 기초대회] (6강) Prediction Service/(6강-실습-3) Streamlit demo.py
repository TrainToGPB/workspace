"""
콘텐츠 라이선스

WARNING : 본 교육 콘텐츠의 지식재산권은 재단법인 네이버커넥트에 귀속됩니다. 본 콘텐츠를 어떠한 경로로든 외부로 유출 및 수정하는 행위를 엄격히 금합니다. 다만, 비영리적 교육 및 연구활동에 한정되어 사용할 수 있으나 재단의 허락을 받아야 합니다. 이를 위반하는 경우, 관련 법률에 따라 책임을 질 수 있습니다.
"""

from datetime import date

import pandas as pd
import requests
import streamlit as st


def streamlit():
    st.title('N2M 데모')
    st.markdown('**Transformers Encoder-Decoder 모델을 활용한 Sequence to Sequence 문제 실습**')
    st.markdown("#### 데이터셋([Link](https://github.com/htw5295/Neural_date_translation_dataset))", unsafe_allow_html=True)
    data_info = """
- Faker 라이브러리로 생성한 날짜 표기 데이터
- 입력 : 다양한 형태의 날짜 표기 데이터
- 출력 : yyyy-mm-dd 형태의 날짜 표기 데이터
- 학습 데이터 : 24,000개
- 검증 데이터 : 3,000개
- 평가 데이터 : 3,000개
    """
    st.markdown(data_info)
    st.markdown("#### 모델")
    model_info = """
- [facebook/bart-base](https://huggingface.co/facebook/bart-base)의 Tokenizer와 Config 활용
- Huggingface의 [AutoModelForSeq2SeqLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSeq2SeqLM) 모델 활용
    """
    st.markdown(model_info)
    st.markdown("#### 평가")
    eval_info = """
- 예측 데이터를 yyyy-mm-dd 형식으로 디코딩한 뒤 정답 데이터와 비교하여 일치, 불일치를 판단함
    """
    st.markdown(eval_info)


    today = date.today().strftime('%b %d %Y')
    st.subheader('Input human readable date text')
    input_date = st.text_input('Input date', today)

    if st.button('Translation'):
        result = requests.get('http://127.0.0.1:8000/translation', params={'date': input_date}).json()

        data = []
        index_data = []
        for i, pred in enumerate(result['preds']):
            data.append([input_date, pred])
            index_data.append(f"Beam {i}")

        df = pd.DataFrame(data, index=index_data, columns=['input date', 'generate date'])

        st.subheader('Generated yyyy-mm-dd format date text')
        st.table(df)

streamlit()
