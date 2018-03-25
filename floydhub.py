from seq2seq import utils, trainer

MODEL_PATH = '/output'
DATASET_PATH = '/input'

hparams = utils.load_hparams('hparams.json')

normalizer_trainer = trainer.Trainer(
    data_dir=DATASET_PATH, model_dir=MODEL_PATH, hparams=hparams)
normalizer_trainer.train()