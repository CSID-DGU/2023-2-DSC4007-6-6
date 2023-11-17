Config = dict(
    MAX_FRAMES = 23,
    EPOCHS = 15,
    LR = 2e-4,
    IMG_SIZE = (224, 224),
    FEATURE_EXTRACTOR = 'efficientnet_lite3', # resnext50_32x4d
    DR_RATE = 0.35,
    NUM_CLASSES = 2, # 바꾸기.... 
    RNN_HIDDEN_SIZE = 100,
    RNN_LAYERS = 1,
    TRAIN_BS = 4,
    VALID_BS = 2,
    NUM_WORKERS = 2,
    #infra = "Kaggle",
    #competition = 'rsna_miccai',
    #_wandb_kernel = 'tanaym'
)

classes={
    'Abuse':1,
    'Arrest':1,
    'Arson':1,
    'Assault':1,
    'Burglary':1,
    'Explosion':1,
    'Fighting':1,
    'RoadAccidents':1,
    'Robbery':1,
    'Shooting':1,
    'Shoplifting':1,
    'Stealing':1,
    'Vandalism':1,
    'Normal':0
    
    
}