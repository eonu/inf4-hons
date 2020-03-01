import numpy as np
from sequentia.preprocessing.transforms import Equalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

class NNClassifier:
    def __init__(self, epochs, batch_size, classes, optimizer='adam'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.classes = classes
        self.optimizer = optimizer
        self.equalizer = Equalize()
        
    def predict(self, X, return_scores=False):
        single = isinstance(X, np.ndarray)
        
        if single:
            X = np.expand_dims(self.equalizer.transform(X), axis=0)
        else:
            X = np.array(self.equalizer.transform(X))
            
        scores = self.model.predict(X)
        preds = [self.classes[i] for i in np.argmax(scores, axis=1)]
        
        if single:
            return (preds[0], scores[0]) if return_scores else preds[0]
        else:
            return [(preds[i], scores[i]) for i in range(len(X))] if return_scores else preds
        
    def evaluate(self, X, y):
        assert isinstance(X, list)
        preds = self.predict(X, return_scores=False)
        cm = confusion_matrix(y, preds, labels=self.classes)
        acc = np.sum(np.diag(cm)) / np.sum(cm)
        return acc, cm
        
    def _one_hot(self, y):
        return np.array([to_categorical(self.classes.index(label), num_classes=len(self.classes)) for label in y])
    
    def summary(self):
        self.model.summary()

class LSTMClassifier(NNClassifier):
    def fit(self, X, y, validation_data=None, verbose=True, return_history=False):
        X, y = np.array(self.equalizer.fit_transform(X)), self._one_hot(y)
        N, T, D = X.shape
        
        # Construct the model
        self.model = Sequential([
            Bidirectional(LSTM(150, return_sequences=True, recurrent_dropout=0.5), input_shape=(T, D)),
            Bidirectional(LSTM(150, recurrent_dropout=0.5)),
            Dense(100, activation='relu'),
            Dropout(0.5),
            Dense(len(self.classes), activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        
        # Fit the model
        if validation_data is None:
            history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)
        else:
            X_val, y_val = validation_data
            X_val, y_val = np.array(self.equalizer.transform(X_val)), self._one_hot(y_val)
            history = self.model.fit(X, y, validation_data=(X_val, y_val), 
                epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)
            
        if return_history:
            return history
        
class FFNNClassifier(NNClassifier):
    def fit(self, X, y, validation_data=None, verbose=True, return_history=False):
        X, y = np.array(self.equalizer.fit_transform(X)), self._one_hot(y)
        N, T, D = X.shape
        
        # Construct the model
        self.model = Sequential([
            Flatten(),
            Dense(150, activation='relu'),
            Dropout(0.5),
            Dense(150, activation='relu'),
            Dropout(0.5),
            Dense(150, activation='relu'),
            Dropout(0.5),
            Dense(len(self.classes), activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        
        # Fit the model
        if validation_data is None:
            history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)
        else:
            X_val, y_val = validation_data
            X_val, y_val = np.array(self.equalizer.transform(X_val)), self._one_hot(y_val)
            history = self.model.fit(X, y, validation_data=(X_val, y_val), 
                epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)
            
        if return_history:
            return history