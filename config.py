import datetime
from utils import dice_coef
from tensorflow.keras.optimizers import Adam
# from wce import weighted_categorical_crossentropy
# loss_func = weighted_categorical_crossentropy([0.0216864 ,  0.38078799,  1.96028943,  1.60354085, 13.47695266, 0.72654349])
# class_weights = {
#     0: 0.021686403607047856,
#     1: 0.380787993075767,
#     2: 1.9602894304075587,
#     3: 1.6035408474802415,
#     4: 13.476952658623624,
#     5: 0.7265434945800464
# }

def get_config():
    return {
        # Network architecture
        "net": {
            "epochs": 1000,
            "verbose": 1,
        },
        "net_cmp": {
            "optimizer": Adam(learning_rate=3e-4), 
            "metrics": [dice_coef]
        },
        # Data paths
        "data": {
            "train_path": 'data/train/',
            "val_path": 'data/val/',
            "test_path": 'data/test/',
            "out_path": 'data/out/',
            "256_train_path": 'data_256_200_64/train_train/',
            "256_val_path": 'data_256_200_64/train_val/',
            "256_test_path": 'data_256_200_64/test/',
            "256_out_path": 'data_256_200_64/out/',
            
        },
        # For checkpoint saving, early stopping...
        "train": {
            "ckpt": {
                "ckpt_path": 'ckpt',
                "verbose": 1, 
                "save_best_only": True
            },
            "early_stopping": {
                "patience": 50, 
                "monitor": 'val_loss'
            }

        }
    }

