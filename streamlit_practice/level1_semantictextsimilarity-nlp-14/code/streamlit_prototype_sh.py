import streamlit as st
import torch
import pytorch_lightning as pl
from sts_train import Dataloader, Dataset
import pandas as pd

st.title("Language Titan 문장 유사도 검사")

sentence1 = st.text_input("첫 번째 문장을 입력해주세요.")
sentence2 = st.text_input("두 번째 문장을 입력해주세요.")


class StreamlitDataloader(Dataloader):
    def setup(self, stage='fit'):
        if stage != 'fit':
            # 평가데이터 준비
            predict_data = pd.DataFrame({'id':'', 'sentence_1':[sentence1], 'sentence_2':[sentence2]})
            predict_inputs, _ = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])


@st.cache_resource
def load_model():
    model = torch.load("sts_model_krelectra.pt")
    trainer = pl.Trainer(accelerator='gpu')
    return model, trainer

if st.button("유사도 측정"):
    with st.spinner("Wait for it..."):
        model, trainer = load_model()

        raw_prediction = trainer.predict(model=model, datamodule=StreamlitDataloader("snunlp/KR-ELECTRA-discriminator", 1, True, "", "", "", ""))[0].item()
        prediction = round(max(min(raw_prediction, 5.0), 0.0), 1)

    st.subheader("두 문장의 STS 유사도:")
    if prediction < 3:
        st.write(str(prediction))
    elif prediction >= 3 and prediction < 4:
        st.subheader(str(prediction))
    elif prediction >=4 and prediction < 5:
        st.header(str(prediction))
    elif prediction == 5:
        st.header(str(prediction) + "!!!")
        st.balloons()