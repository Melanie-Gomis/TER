# Configuration minimale pour le modèle multi-modal

import torch

CONFIG = {
    # ------------------------
    # DEVICE / REPRODUCTIBILITÉ
    # ------------------------
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "SEED": 42,
    "BATCH_SIZE": 32,  
    "NUM_WORKERS": 2,  
    "PIN_MEMORY": True,  

    # ------------------------
    # DROPOUT / REGULARISATION
    # ------------------------
    "MODALITY_DROP_OUT": 0.2,            # fraction des modalités à couper
    "MOD_DO_PROBA": 0.3,                 # probabilité de dropout ciblé
    "SPECEFIC_MODALITIES_SWITCH_OFF": None,
    "VAL_MODALITIES_TO_SWITCH_OFF": None,

    # ------------------------
    # CNN / MODEL
    # ------------------------
    "BATCH_NORM": True,                  # BatchNorm dans BasicConv2d
    "USE_CNN_ADV": False,                # Récupération représentation latente CNN avancée
    "ONE_CONV_AFTER_INCEPTIONS": False,  # simplifie les convolutions après Inception
    "USE_CROSS_FUSION": False,           # fusion croisée des modalités
    "USE_MODALITY_TRANSFORMERS": False,  # transformer sur dimension 5
    "CNN_INPUT_STAGE": 1,                # étape pour récupérer représentation latente CNN

    # ------------------------
    # DATA
    # ------------------------
    "INPUT_SHAPE": (64, 64, 9),
    "N_MODALITIES": 9,
}
