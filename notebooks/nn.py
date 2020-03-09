import numpy as np
from sequentia.preprocessing.transforms import Equalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix

class NNClassifier:
    def __init__(self, epochs, batch_size, classes, optimizer='adam'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.classes = classes
        self.optimizer = optimizer
        self.equalizer = Equalize()
        
    def fit(self, architecture, X, y, validation_data, verbose=2, return_history=False, 
            early_stop=True, patience=None, checkpoint=False, checkpoint_path=None):
        X, y = np.array(self.equalizer.fit_transform(X)), self._one_hot(y)
        
        self.model = Sequential(architecture)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        
        X_val, y_val = validation_data
        X_val, y_val = np.array(self.equalizer.transform(X_val)), self._one_hot(y_val)
        
        callbacks = []
        if early_stop:
            assert patience is not None
            es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=(verbose > 1), patience=patience)
            callbacks.append(es)
        if checkpoint:
            assert checkpoint_path is not None
            mc = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', mode='max', verbose=(verbose > 1), save_best_only=True)
            callbacks.append(mc)
            
        history = self.model.fit(X, y, validation_data=(X_val, y_val), 
            epochs=self.epochs, batch_size=self.batch_size, verbose=(verbose > 0), callbacks=callbacks)
            
        if return_history:
            return history
        
    def predict(self, X, return_scores=False):
        single = isinstance(X, np.ndarray)
        transformed = self.equalizer.transform(X)
        
        X = np.expand_dims(transformed, axis=0) if single else np.array(transformed)
            
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