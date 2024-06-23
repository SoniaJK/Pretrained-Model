from src.backbone import TFLiteModel, get_model

def load_model():
    model_path = '/Pretrained-Model/gesture_project/models/islr-fp16-192-8-seed42-fold0-best.h5'  # Adjust the path if necessary
    model = get_model()
    model.load_weights(model_path)
    return model
