from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from typing import Optional
from sentencepiece import SentencePieceProcessor as spm
from loguru import logger
import wandb


class TrainModel:
    def __init__(self, 
        model, # Modelo a ser treinado
        train_loader: DataLoader, # Dataloader de treino
        val_loader: DataLoader,
        lerning_rate: float = 1e-4, # Taxa de aprendizado
        weight_decay: float = 1e-2, # Peso da regularização L2
        tokenizer: Optional[spm] = None, # Tokenizador
        num_epochs: int = 1, # Número de épocas
        eval_freq: int = 5, # Frequência de avaliação
        num_eval_batchs: int = 10, # Número de batchs usados para avaliação
        start_context: str = "O Brasil é um", # Contexto inicial para geração de texto
    ):
        self.MODEL = model
        self.TRAIN_LOADER = train_loader
        self.VAL_LOADER = val_loader
        self.LR = lerning_rate
        self.WD = weight_decay
        self.NUM_EPOCHS = num_epochs
        self.EVAL_FREQ = eval_freq
        self.NUM_EVAL_BATCHS = num_eval_batchs
        self.START_CONTEXT = start_context
        self.TOKENIZER = tokenizer

        self.OPTIMIZER = None
        self.DEVICE = None
        self.HISTORY = {}
        self.GLOBAL_STEP = -1
        self.BEST_VAL_LOSS = float('inf')

        self._prepare()

    def _prepare(self):
        logger.info("Preparando modelo para treinamento")
        self.DEVICE = self._get_device()
        self.MODEL.to(self.DEVICE)
        self.MODEL.train()

        self.GLOBAL_STEP = -1
        self.HISTORY = {
            'train_loss': [], 'val_loss': [], 
            'train_ppl': [], 'val_ppl': [],
            'train_val_diff': [], 'grad_norm': [],
            'lr': [],
            'tokens_seen': []
        }

        p_dict = {p_name: p for p_name, p in self.MODEL.named_parameters() if p.requires_grad}
        weight_decay_p = [p for n, p in p_dict.items() if p.dim() >= 2]
        no_weight_decay_p = [p for n, p in p_dict.items() if p.dim() < 2]

        optimizer_groups = [
            {"params": weight_decay_p, "weight_decay": 0.01},
            {"params": no_weight_decay_p, "weight_decay": 0.0}
        ]

        self.OPTIMIZER = torch.optim.AdamW(
            optimizer_groups,
            lr=self.LR,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        self.SCHEDULER = OneCycleLR(
            self.OPTIMIZER,
            max_lr=self.LR,
            total_steps= len(self.TRAIN_LOADER) * self.NUM_EPOCHS,
            pct_start=0.1,  # 10% dos steps para warmup
            div_factor=10,  # lr_inicial = max_lr/10
            final_div_factor=100  # lr_final = lr_inicial/100
        )

        model_config = {
            "model_type": self.MODEL.__class__.__name__,
            "params": sum(p.numel() for p in self.MODEL.parameters()),
            "optimizer": self.OPTIMIZER.__class__.__name__,
            "scheduler": self.SCHEDULER.__class__.__name__,
            "learning_rate": self.OPTIMIZER.param_groups[0]['lr'],
            "batch_size": self.TRAIN_LOADER.batch_size,
            "device": self.DEVICE.type,
            "epochs": self.NUM_EPOCHS,
            "eval_freq": self.EVAL_FREQ,
            "eval_batchs": self.NUM_EVAL_BATCHS
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"run_{timestamp}"

        # Inicializar wandb
        wandb.init(project="LLM_ZERO", name=experiment_name, config=model_config)
        
        # Observar modelo com wandb
        wandb.watch(self.MODEL, log="all", log_freq=self.EVAL_FREQ*10)

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        
        if torch.mps.is_available():
            return torch.device('mps')
        
        return torch.device('cpu')

    def _loss_batch(self, input_batch, target_batch):
        input_batch, target_batch = input_batch.to(self.DEVICE, non_blocking=True), target_batch.to(self.DEVICE, non_blocking=True)

        logits = self.MODEL(input_batch)
        logits_flat = logits.flatten(0, 1)
        target_flat = target_batch.flatten()
        return torch.nn.functional.cross_entropy(logits_flat, target_flat)

    def _loss_loader(self, data_loader):
        total_loss = 0.
        num_batchs = self.NUM_EVAL_BATCHS

        # Verifica se o dataloader está vazio
        if len(data_loader) == 0:
            return float("nan")
        # Se num_batchs for None, calcula a loss para todos os batchs
        elif num_batchs is None:
            num_batchs = len(data_loader)
        # Se num_batchs for maior que o tamanho do dataloader, calcula a loss para todos os batchs do data loader
        else:
            num_batchs = min(num_batchs, len(data_loader))

        for i, (x, y) in enumerate(data_loader):
            if i < num_batchs:
                loss = self._loss_batch(x, y)
                total_loss += loss.item()
            else:
                break

        return total_loss / num_batchs

    def _calculate_gradient_norm(self):
        total_norm = 0.0
        for p in self.MODEL.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def _eval_model(self):
        was_training = self.MODEL.training
        self.MODEL.eval()
        results = {}
        
        with torch.no_grad():
            train_loss = self._loss_loader(self.TRAIN_LOADER)
            val_loss = self._loss_loader(self.VAL_LOADER)
            
            train_ppl = torch.exp(torch.tensor(train_loss)).item()
            val_ppl = torch.exp(torch.tensor(val_loss)).item()
            train_val_diff = val_loss - train_loss
        
        results['train_loss'] = train_loss
        results['val_loss'] = val_loss
        results['train_ppl'] = train_ppl
        results['val_ppl'] = val_ppl
        results['train_val_diff'] = train_val_diff
        
        if was_training:
            self.MODEL.train()
            
        return results

    @torch.no_grad()
    def _generate_sample(self, max_token=200, temperature=1, top_k=0, top_p=1):
        input_tokens = torch.tensor(self.TOKENIZER.encode(self.START_CONTEXT), dtype=torch.long)
        input_tokens = input_tokens.unsqueeze(0).to(self.DEVICE)

        output = self.MODEL.generate(input_tokens, max_token, temperature, top_k, top_p)
        output_text = self.TOKENIZER.decode(output.squeeze().tolist())

        return output_text

    def train(self):
        logger.info("Iniciando treinamento")
        tokens_seen = 0

        for epoch in range(self.NUM_EPOCHS):
            self.MODEL.train()

            for x_batch, y_batch in self.TRAIN_LOADER:

                self.OPTIMIZER.zero_grad()
                loss = self._loss_batch(x_batch, y_batch)
                loss.backward()
                
                # Capturar a norma do gradiente atual
                current_grad_norm = float('nan')
                try:
                    current_grad_norm = self._calculate_gradient_norm()
                except Exception as e:
                    logger.error(f"Erro ao calcular norma do gradiente: {e}")
                
                torch.nn.utils.clip_grad_norm_(self.MODEL.parameters(), max_norm=1.0)
                self.OPTIMIZER.step()
                self.SCHEDULER.step()

                tokens_seen += x_batch.numel()
                self.GLOBAL_STEP += 1
                
                if self.GLOBAL_STEP % self.EVAL_FREQ == 0:
                    
                    eval_results = self._eval_model()
                    eval_results['grad_norm'] = current_grad_norm
                    eval_results['tokens_seen'] = tokens_seen
                    eval_results['lr'] = self.SCHEDULER.get_last_lr()[0]
                    
                    for key, value in eval_results.items():
                        if key in self.HISTORY:
                            self.HISTORY[key].append(value)

                    # Log das métricas principais
                    log_dict = {k: v for k, v in eval_results.items()}
                    wandb.log(log_dict, step=self.GLOBAL_STEP)
                    
                    # Log de informações adicionais
                    logger.debug(f"Epoch {epoch + 1} [Step {self.GLOBAL_STEP:04d}]\nTrain loss: {eval_results['train_loss']: 4f} | Val loss: {eval_results['val_loss']:.4f} | Diff: {eval_results['train_val_diff']:.4f} | Val PPL: {eval_results['val_ppl']:.2f} | Gradient norm: {current_grad_norm:.6f} | Tokens processados: {tokens_seen:,}")

                    # Salvar checkpoint
                    if eval_results['val_loss'] < self.BEST_VAL_LOSS:
                        logger.info(f"Savando checkpoint - Val loss: {eval_results['val_loss']:.4f}")
                        
                        self.BEST_VAL_LOSS = eval_results['val_loss']
                        checkpoint = {
                            'model_state_dict': self.MODEL.state_dict(),
                            'optimizer_state_dict': self.OPTIMIZER.state_dict(),
                            'scheduler_state_dict': self.SCHEDULER.state_dict(),
                            'history': self.HISTORY,
                            'global_step': self.GLOBAL_STEP
                        }
                        torch.save(checkpoint, "models/pretrain/checkpoint.pth")
                    
                    # Geração de texto periódica
                    if self.GLOBAL_STEP % (self.EVAL_FREQ * 20) == 0:
                        try:
                            sample = self._generate_sample(
                                max_token=50, 
                                temperature=0.8
                            )
                            logger.debug(f"Sample: {sample}")
                        except Exception as e:
                            print(f"Erro na geração de texto: {e}")
