import torch
from dataloader import create_dataloader
from loguru import logger
from train_model import TrainModel
from model import GPTModel
import sentencepiece as spm


tk = spm.SentencePieceProcessor(
    model_file="models/gigaverbo_tk.model"
)

def prepare_dataset():

    logger.info("Preparando dataset para treinamento")
    with open("data/gigaverbo.txt", "r", encoding="utf-8") as f:
        text = f.read()

    logger.info("Fazendo Split do dataset")
    train_ratio = 0.9
    split_idx = int(len(text) * train_ratio)

    train_data = text[:split_idx]
    val_data = text[split_idx:]

    torch.manual_seed(123)

    logger.info("Criando dataloader de treinamento")
    train_loader = create_dataloader(
        train_data,
        tk,
        batch_size=32,
        max_length=256,
        stride=128,
        drop_last=True,
        shuffle=True,
        num_workers=4
    )

    logger.info("Criando dataloader de validação")
    val_loader = create_dataloader(
        val_data,
        tk,
        batch_size=32,
        max_length=256,
        stride=128,
        drop_last=True,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader

def create_model():
    logger.info("Criando modelo GPT")

    MODEL = GPTModel(
        d_vocab=tk.vocab_size(),
        d_emb=768,
        context_length=256,
        n_layers=12,
        n_heads=12,
        dropout=0.1,
        qkv_bias=True,
    )

    return MODEL


if __name__ == "__main__":

    train_loader, val_loader = prepare_dataset()
    model = create_model()

    trainer = TrainModel(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lerning_rate=3e-4,
        weight_decay=5e-3,
        tokenizer=tk,
        num_epochs=1,
        eval_freq=5,
        num_eval_batchs=10,
        start_context="O Brasil é um",
    )

    try:
        trainer.train()

    except KeyboardInterrupt:
        del trainer
        del model
        del train_loader
        del val_loader

        torch.mps.empty_cache() if torch.mps.is_available() else None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        logger.info("Treinamento interrompido")