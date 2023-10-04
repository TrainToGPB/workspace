"""
콘텐츠 라이선스

WARNING : 본 교육 콘텐츠의 지식재산권은 재단법인 네이버커넥트에 귀속됩니다. 본 콘텐츠를 어떠한 경로로든 외부로 유출 및 수정하는 행위를 엄격히 금합니다. 다만, 비영리적 교육 및 연구활동에 한정되어 사용할 수 있으나 재단의 허락을 받아야 합니다. 이를 위반하는 경우, 관련 법률에 따라 책임을 질 수 있습니다.
"""

from fastapi import FastAPI
import uvicorn
from starlette.responses import JSONResponse
from temp.date import Model

app = FastAPI()


@app.get("/translation")
async def root(date):
    inputs = model.tokenizer(date, padding='max_length', truncation=True, max_length=16, return_tensors='pt')
    pred_ids = model.encoder_decoder.generate(inputs['input_ids'], num_beams=3, min_length=0, max_length=16, num_return_sequences=3)
    pred = model.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return JSONResponse(content={"preds": pred})


if __name__ == '__main__':
    # 아래 경로에 캠퍼분들의 저장된 모델 경로를 입력하시면 됩니다.
    model = Model.load_from_checkpoint('/workspace/temp/lightning_logs/version_16/checkpoints/epoch=0-step=188.ckpt')

    uvicorn.run(app, host="0.0.0.0", port=8000)
