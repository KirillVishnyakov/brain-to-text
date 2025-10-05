from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

import warnings

warnings.filterwarnings("ignore")

args = OmegaConf.load('rnn_args.yaml')
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()