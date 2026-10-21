"""Configuration defaults for the chest X-ray multi-label project."""

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

NUM_CLASSES = len(LABELS)
DATASET_NAME = "NIH ChestX-ray14 sample"
TASK_TYPE = "multi-label classification"
