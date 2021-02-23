import tensorflow


def load_classification_model(model_folder):
    classification_model = tensorflow.keras.models.load_model(model_folder)

    # Check its architecture
    # print(classification_model.summary())

    return classification_model
