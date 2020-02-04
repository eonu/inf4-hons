import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

class LSTMClassifier:
    def __init__(self, epochs, batch_size, classes, optimizer='adam'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.classes = classes
        self.optimizer = optimizer
    
    def fit(self, X, y, validation_data=None, verbose=True, return_history=False):
        self.T = max(len(x) for x in X)
        X, y = self._transform(X, y)
        
        # N: Number of training examples
        # T: Number of frames (truncated/padded)
        # D: Dimensionality of training examples
        N, T, D = X.shape
        
        # Construct the model
        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=(T, D)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(len(self.classes), activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        
        # Fit the model
        if validation_data is None:
            history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)
        else:
            history = self.model.fit(X, y, validation_data=self._transform(*validation_data), 
                epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)
            
        if return_history:
            return history
        
    def predict(self, X, return_scores=False):
        single = isinstance(X, np.ndarray)
        X = self._transform([X] if single else X)
        scores = self.model.predict(X)
        preds = [self.classes[i] for i in np.argmax(scores, axis=1)]
        output = [(preds[i], scores[i]) for i in range(len(X))] if return_scores else preds
        return output[0] if single else output
        
    def evaluate(self, X, y):
        preds = self.predict(X)
        cm = confusion_matrix(y, preds, labels=self.classes)
        acc = np.sum(np.diag(cm)) / np.sum(cm)
        return acc, cm
        
    def _transform(self, X, yy=None):
        def transform_x(x):
            # Zero-padding or removal to ensure inputs are all the same length
            T, D = x.shape
            return np.vstack((x, np.zeros((self.T - T, D)))) if T <= self.T else x[:self.T]
        def transform_y(y):
            # Generate one-hot encodings
            return to_categorical(self.classes.index(y), num_classes=len(self.classes))
        if yy is None:
            return np.array([transform_x(x) for x in X])
        else:
            return np.array([transform_x(x) for x in X]), np.array([transform_y(y) for y in yy])