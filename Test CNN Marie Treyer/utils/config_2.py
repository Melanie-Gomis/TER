# Configuration minimale pour le modèle multi-modal

import torch

CONFIG = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "SEED": 42,
    "BATCH_SIZE": 32,
    "NUM_WORKERS": 2,
    "PIN_MEMORY": True,

    # régularisation
    "MODALITY_DROP_OUT": None,   # maintenant utile
    "MOD_DO_PROBA": 0.3,
    "SPECEFIC_MODALITIES_SWITCH_OFF": None,
    "VAL_MODALITIES_TO_SWITCH_OFF": None,

    # modèle
    "BATCH_NORM": True,
    "USE_CNN_ADV": False,
    "ONE_CONV_AFTER_INCEPTIONS": False,
    "USE_CROSS_FUSION": False,
    "USE_MODALITY_TRANSFORMERS": False,
    "CNN_INPUT_STAGE": 1,

    # data
    "INPUT_SHAPE": (64, 64, 9),
}