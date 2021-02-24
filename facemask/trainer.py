# imports
from sklearn.model_selection import train_test_split
from facemask.data import get_data, dataset_preproc, create_dataframe
import joblib
from google.cloud import storage
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.model = None
        self.X = X
        self.y = y

    def initialize_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3),padding='same'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu',padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu',padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(3,3)))
        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(60, activation='relu'))
        model.add(layers.Dropout(rate=0.5))
        model.add(layers.Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model


    def run(self):
        """set and train the model"""
        model=self.initialize_model()
        es = EarlyStopping(patience=10,restore_best_weights=True)

        history = self.model.fit(self.X, self.y,
                    validation_split=0.3,
                    epochs=500,
                    batch_size=32,
                    verbose=0,
                    callbacks=[es])

    def evaluate(self, X_test, y_test):
        """evaluates the model on df_test and return the accuracy"""

        accuracy_model=self.model.evaluate(X_test, y_test, verbose=0)[1]
        return accuracy_model


    def save_model(self):
        # using keras method
        self.model.save('models/model1_300')
        # joblib.dump(self.model, 'model1_300.joblib')
        print("saved model1_6000.joblib locally")

        # storage_client = storage.Client()
        # bucket = storage_client.bucket('')
        # blob = bucket.blob('models/model_v1')

        # blob.upload_from_filename('model.joblib')


        # print(f"uploaded model.joblib to gcp cloud storage")

if __name__ == "__main__":
    # get X and y
    X1,X2,X3 = get_data()
    X, y = create_dataframe(X1,X2,X3)
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train_norm, y_train_cat = dataset_preproc(X_train, y_train)
    X_test_norm, y_test_cat = dataset_preproc(X_test, y_test)
    # train
    trainer = Trainer(X_train_norm, y_train_cat)

    trainer.run()
    # evaluate
    accuracy_model=trainer.evaluate(X_test_norm, y_test_cat)
    print(f'The accuracy is: {accuracy_model}')
    trainer.save_model()
