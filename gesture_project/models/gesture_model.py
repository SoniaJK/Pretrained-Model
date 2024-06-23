from src.backbone import TFLiteModel, get_model

def load_model():
    model_path = './model/your_model.h5'  # Adjust the path if necessary
    model = get_model()
    model.load_weights(model_path)
    return model