import datetime
from utils import dice_coef

def get_config():
    return {
        # Network architecture
        "net": {
            "epochs": 1000,
            "verbose": 1,
        },
        "net_cmp": {
            "optimizer": 'adam', 
            "loss": 'categorical_crossentropy', 
            "metrics": ['accuracy', dice_coef]
        },
        # Data paths
        "data": {
            "train_path": 'data/train/',
            "val_path": 'data/val/',
            "test_path": 'data/test/'
        },
        # For checkpoint saving, early stopping...
        "train": {
            "ckpt": {
                "ckpt_path": 'ckpt',
                "verbose": 1, 
                "save_best_only": True
            },
            "early_stopping": {
                "patience": 10, 
                "monitor": 'val_loss'
            },
            "tensorboard": {
                "log_dir": "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                "histogram_freq": 1
            }

        }
    }

