from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer


args = OmegaConf.load('model_training/rnn_args.yaml')
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()