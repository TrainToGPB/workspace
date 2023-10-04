import pandas as pd
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm

import torch
import pytorch_lightning as pl

import transformers


# https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])

    def __len__(self):
        return len(self.data)

# https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html?highlight=datamodule
class Dataloader(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # 페이스북 bart-base 모델의 토크나이저 불러오기
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/bart-base')

        # github에 업로드된 데이터셋 다운로드
        # index 컬럼이 없으므로 index_col= False
        # 데이터를 토큰화
        self.train_data = self.tokenizing(pd.read_csv('https://raw.githubusercontent.com/htw5295/Neural_date_translation_dataset/main/train.csv', index_col=False))
        self.val_data   = self.tokenizing(pd.read_csv('https://raw.githubusercontent.com/htw5295/Neural_date_translation_dataset/main/val.csv', index_col=False))
        self.test_data  = self.tokenizing(pd.read_csv('https://raw.githubusercontent.com/htw5295/Neural_date_translation_dataset/main/test.csv', index_col=False))

    def tokenizing(self, dataframe):
        tokenized_data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='text tokenizing', total=len(dataframe)):
            # 최대길이 16, truncation -> 최대길이에 맞게 데이터 자르기, padding -> 최대길이에 맞게 패딩토큰 추가하기
            input_date  = self.tokenizer(item['inputs'],  padding='max_length', truncation=True, max_length=16)
            target_date = self.tokenizer(item['targets'], padding='max_length', truncation=True, max_length=16)
            # 텍스트를 숫자로 변환한 input_ids를 리스트 형식으로 저장
            tokenized_data.append([input_date['input_ids'], target_date['input_ids']])

        return tokenized_data

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = Dataset(self.train_data)
            self.val_dataset   = Dataset(self.val_data)
        else:
            self.test_dataset  = Dataset(self.test_data)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=32)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

# https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
class Model(pl.LightningModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = tokenizer

        # 페이스북 bart-base 모델의 설정값 불러오기
        self.config = transformers.BartConfig.from_pretrained('facebook/bart-base')

        # 불러온 설정값을 토대로 AutoModelForSeq2SeqLM(BART) 모델 생성
        self.encoder_decoder = transformers.AutoModelForSeq2SeqLM.from_config(self.config)

    def forward(self, x, y):
        outputs  = self.encoder_decoder(input_ids=x, labels=y)
        return outputs.loss, outputs.logits

    def training_step(self, batch, batch_idx):
        x, y = batch

        # loss 측정
        loss, logits = self(x, y)
        self.log("train_loss", loss)

        # 훈련/검증 단계 때는 LOSS 등만 확인하지만,
        # 최종 예측은 beam search 를 통해 수행되기 때문에
        # 훈련 중간 중간의 결과물을 확인하기 위해서 아래 코드를 사용할 수 있습니다.
        # (일반적으로는 속도가 매우 느려지기 때문에 훈련단계에서는 포함하지 않습니다)

        # Beam search의 N=3으로, 3개의 문장을 생성하고, 가장 좋은 1개의 문장을 받아옴
        pred_ids = self.encoder_decoder.generate(x, num_beams=3, min_length=0, max_length=16, num_return_sequences=1)
        # 토큰 -> 텍스트 변환
        pred     = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        target   = self.tokenizer.batch_decode(y, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # 예측값과 정답값이 일치하는지 측정
        accuracy = []
        for p, t in zip(pred, target):
            if p == t:
                accuracy.append(1)
            else:
                accuracy.append(0)
        accuracy = sum(accuracy) / len(accuracy)
        self.log("train_acc", accuracy, prog_bar=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # loss 측정
        loss, logits = self(x, y)
        self.log("val_loss", loss)

        # 훈련/검증 단계 때는 LOSS 등만 확인하지만,
        # 최종 예측은 beam search 를 통해 수행되기 때문에
        # 훈련 중간 중간의 결과물을 확인하기 위해서 아래 코드를 사용할 수 있습니다.
        # (일반적으로는 속도가 매우 느려지기 때문에 훈련단계에서는 포함하지 않습니다)

        # 예측값과 정답값이 일치하는지 비교하기 위해 예측 토큰 생성
        # Beam search의 N=3으로, 3개의 문장을 생성하고, 가장 좋은 1개의 문장을 받아옴
        pred_ids = self.encoder_decoder.generate(x, num_beams=3, min_length=0, max_length=16, num_return_sequences=1)
        # 토큰 -> 텍스트 변환
        pred = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        target = self.tokenizer.batch_decode(y, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # 예측값과 정답값이 일치하는지 측정
        accuracy = []
        for p, t in zip(pred, target):
            if p == t:
                accuracy.append(1)
            else:
                accuracy.append(0)
        accuracy = sum(accuracy) / len(accuracy)
        self.log("val_acc", accuracy, prog_bar=True, on_step=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        # 예측값과 정답값이 일치하는지 비교하기 위해 예측 토큰 생성
        # Beam search의 N=3으로, 3개의 문장을 생성하고, 가장 좋은 1개의 문장을 받아옴
        pred_ids = self.encoder_decoder.generate(x, num_beams=3, min_length=0, max_length=16, num_return_sequences=1)
        # 토큰 -> 텍스트 변환
        pred = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        target = self.tokenizer.batch_decode(y, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # 예측값과 정답값이 일치하는지 측정
        accuracy = []
        for p, t in zip(pred, target):
            if p == t:
                accuracy.append(1)
            else:
                accuracy.append(0)
        accuracy = sum(accuracy) / len(accuracy)
        self.log("test_acc", accuracy, prog_bar=True)

        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer


if __name__ == '__main__':
    batch_size = 128
    max_epoch = 1

    # W&B 로그 설정, 생성한 프로젝트 이름 입력
    # https://wandb.ai/{유저닉네임}/{프로젝트이름} 에서 확인가능
    wandb_logger = WandbLogger(project="date")

    dataloader = Dataloader(batch_size)
    model = Model(dataloader.tokenizer)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="model", save_top_k=1, monitor="val_loss")
    # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    trainer = pl.Trainer(gpus=1, max_epochs=max_epoch, logger=wandb_logger, callbacks=[checkpoint_callback], log_every_n_steps=1)
    trainer.fit(model=model, datamodule=dataloader)

    trainer.test(model=model, datamodule=dataloader)  # <- test set 에 대한 평가 진행됨
