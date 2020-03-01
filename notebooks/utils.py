import os, glob, re, pathlib
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sequentia.preprocessing import Transform, Downsample
from tqdm.auto import tqdm

__all__ = ['data_split', 'BinDownsample', 'show_results', 'show_class_counts', 'show_durations', 'show_accuracy_history', 'show_loss_history', 'write_knn_results', 'write_hmm_results', 'write_network_results', 'MoCapLoader']

# ggplot style
plt.style.use('ggplot')

def savefig(save):
    if save is not None:
        plt.savefig(os.path.join('Plots', save))

def data_split(X, y, splits, random_state=None, stratify=False):
    """Generate a training, validation and test dataset split"""
    X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []
    
    splits = [s / 100. for s in splits]
    assert sum(splits) == 1.
    
    rest_size = 1. - splits[0]
    test_size = splits[2] / rest_size
    
    if stratify:
        outer_splitter = StratifiedShuffleSplit(n_splits=1, test_size=rest_size, random_state=random_state)
        for train_idx, rest_idx in outer_splitter.split(X, y):
            X_train, X_rest = [X[i] for i in train_idx], [X[i] for i in rest_idx]
            y_train, y_rest = [y[i] for i in train_idx], [y[i] for i in rest_idx]
            inner_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            for val_idx, test_idx in inner_splitter.split(X_rest, y_rest):
                X_val, X_test = [X_rest[i] for i in val_idx], [X_rest[i] for i in test_idx]
                y_val, y_test = [y_rest[i] for i in val_idx], [y_rest[i] for i in test_idx]
    else:
        X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=rest_size, shuffle=True, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=test_size, shuffle=True, random_state=random_state)
        
    # Display the split sizes
    print('Training set size: {}'.format(len(y_train)))
    print('Validation set size: {}'.format(len(y_val)))
    print('Test set size: {}'.format(len(y_test)))
        
    return X_train, X_val, X_test, y_train, y_val, y_test

class BinDownsample(Transform):
    def __init__(self, bin_size, method='decimate'):
        super().__init__()
        self.bin_size = bin_size
        self.method = self._val.one_of(method, ['decimate', 'mean'], desc='downsampling method')

    def _describe(self):
        method = 'Decimation' if self.method == 'decimate' else 'Mean'
        return '{} bin-downsampling with bin-size {}'.format(method, self.bin_size)

    def transform(self, X, verbose=False):
        def bin_downsample(x):
            T = len(x)
            factor = int((T - T % -self.bin_size) / self.bin_size) - 1
            return x if factor == 0 else Downsample(factor=factor, method=self.method).transform(x)
        return self._apply(bin_downsample, X, verbose)

def show_results(acc, cm, dataset, labels):
    """Display accuracy and confusion matrix"""
    df = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df, annot=True)
    plt.title('Confusion matrix for {} set predictions'.format(dataset), fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # Fix for matplotlib bug that cuts off top/bottom of seaborn visualizations
    b, t = plt.ylim()
    plt.ylim(b + 0.5, t - 0.5)
    plt.show()
    print('Accuracy: {:.2f}%'.format(acc * 100))
    
def show_class_counts(y, classes, xtick_rotation=0, title=None, figsize=(8, 6), save=None):
    """Display class counts for a dataset"""
    xs = [i for i in range(len(classes))]
    counts = [y.count(label) for label in classes]
    if title is not None:
        title = 'Dataset class counts' if title == 'default' else title
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.bar(xs, counts, color='green')
    tick_align = 'center' if xtick_rotation % 90 == 0 else 'right'
    plt.xticks(xs, classes, rotation=xtick_rotation, ha=tick_align)
    plt.tight_layout()
    savefig(save)
    plt.show()
    
def show_durations(X, bins=None, title=None, figsize=(8, 6), save=None):
    """Display a histogram of sequence durations"""
    durations = [len(x) for x in X]
    if title is not None:
        title = 'Histogram of sequence durations' if title == 'default' else title
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel('Duration (frames)')
    plt.ylabel('Count')
    plt.hist(durations, bins=bins)
    plt.tight_layout()
    savefig(save)
    plt.show()
    
def show_accuracy_history(network, history):
    """Display accuracy history for NN training"""
    hist = history.history
    keys = hist.keys()
    plt.figure(figsize=(8, 6))
    plt.title('{} accuracy history during training'.format(network))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    if 'accuracy' in keys:
        plt.plot(hist['accuracy'], label='training')
    if 'val_accuracy' in keys:
        plt.plot(hist['val_accuracy'], label='validation')
    plt.legend()
    plt.show()
    
def show_loss_history(network, history):
    """Display cross entropy loss history for NN training"""
    hist = history.history
    keys = hist.keys()
    plt.figure(figsize=(8, 6))
    plt.title('{} loss history during training'.format(network))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if 'loss' in keys:
        plt.plot(hist['loss'], label='training')
    if 'val_loss' in keys:
        plt.plot(hist['val_loss'], label='validation')
    plt.legend()
    plt.show()
    
def write_knn_results(results, dataset, name, split, number=None, save_cm=False):
    path = os.path.join('Experiments', dataset, 'knn', name + ' ' + split)
    if number is not None:
        path = path + ' ' + str(number)
    acc, cm = results['knn'][split]
    with open(path, 'w') as file:
        file.write(str(acc))
    if save_cm:
        np.save(path, cm)
        
def write_hmm_results(results, dataset, name, split, number=None, save_cm=False):
    path = os.path.join('Experiments', dataset, 'hmm', '{} {}'.format(name, split))
    if number is not None:
        path = path + ' ' + str(number)
    acc, cm = results['hmm'][split]
    with open(path, 'w') as file:
        file.write(str(acc))
    if save_cm:
        np.save(path, cm)
        
def write_network_results(network, results, dataset, name, split, history=None, number=None, save_cm=False):
    path = os.path.join('Experiments', dataset, network, '{} {}'.format(name, split))
    if number is not None:
        path = path + ' ' + str(number)
    acc, cm = results[network][split]
    with open(path, 'w') as file:
        file.write(str(acc))
    if save_cm:
        np.save(path, cm)
    if history is not None:
        history_path = '{} history.csv'.format(path)
        header = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
        values = np.array(list(history.history.values())).T
        np.savetxt(history_path, values, "%f,%f,%f,%f", header=','.join(header), comments="")
    
# Motion capture dataset utilities
    
class MoCapRecording:
    def __init__(self, csv):
        self.id = pathlib.Path(csv).stem.split('.')[0]
        self.speaker = self.id.split('_')[0]
        self.personality = self.id.split('_')[-1]
        self.path = csv

class MoCapParameters(MoCapRecording):
    def __init__(self, csv, fields):
        self.fields = fields
        super().__init__(csv)
    
    def as_df(self) -> pd.DataFrame:
        """Creates a dataframe from the rov.csv parameters file."""
        # Load the annotations
        df = pd.read_csv(self.path).astype(np.float64) 
        # Select only the rotation vectors
        df = df[self.fields]
        # Add recording identifier column
        df['recording'] = self.id
        return df

class MoCapAnnotations(MoCapRecording):
    def __init__(self, csv, fields, normalized):
        self.params = MoCapParameters(
            re.sub('/rov.csv', '/Normalized/rov.csv' if normalized else '/Original/rov.csv', 
                   re.sub('eaf.csv', 'rov.csv', 
                          re.sub('annotations', 'params', csv))),
            fields
        )
        self.lag = pd.read_csv('../lag.csv', index_col=0, squeeze=True).to_dict()
        super().__init__(csv)     
    
    def as_df(self) -> pd.DataFrame:
        """Creates a dataframe from the eaf.csv annotations file."""
        # Load the annotations
        df = pd.read_csv(self.path)   
        # Select the gesture type, start-time, end-time and duration columns
        df = df[['type', 'start_time', 'end_time', 'during_time']]
        # Rename 'during_time' column to 'duration'
        df = df.rename(columns={'type': 'gesture', 'during_time': 'duration'})
        # Convert the 'duration' column to an integer
        df = df.astype({'start_time': 'int32', 'end_time': 'int32', 'duration': 'int32'})
        # Convert the units from milliseconds to frames (/1000 and *100)
        df.loc[:, ['start_time', 'end_time', 'duration']] //= 10
        # Subtract the lag time from the start and end times
        df[['start_time', 'end_time']] -= round(self.lag[self.id] * 100)
        # Add recording identifier column
        df['recording'] = self.id
        # Reorder columns such that 'gesture' is the last column
        df = df[['start_time', 'end_time', 'duration', 'recording', 'gesture']]
        # Remove 'start' gestures
        df = df[df['gesture'] != 'start']
        return df
    
class MoCapLoader: 
    def __init__(self, normalized=False):
        self.normalized = normalized
        
    def load(self, fields):
        # File paths to all annotation CSV files
        csvs = glob.glob('../annotations/*/eaf.csv/*.eaf.csv')

        # Initialize Annotations object for each eaf.csv annotations file
        # NOTE: Skip the ones I annotated - without a lag measure
        anns = [MoCapAnnotations(csv, fields, self.normalized) for csv in csvs if all(
            skip not in csv for skip in ['sophie_04_e', 'sophie_05_i']
        )]
        
        # Combine the parameters dataframes for each Annotations object
        param_df = pd.concat(ann.params.as_df() for ann in anns)

        # Combine the dataframes for each Annotations object
        ann_df = pd.concat(ann.as_df() for ann in anns)

        # Shuffle the annotations dataframe
        ann_df = ann_df.reset_index(drop=True)
        
        X, y = [], []
        # Convert dataframes to Numpy arrays and skip blank gestures
        # NOTE: Blank gestures may be caused by a larger problem, like recording misalignment - look into this!
        for i, (_, row) in tqdm(enumerate(ann_df.iterrows()), total=len(ann_df), desc='Converting dataframes to Numpy arrays', ncols='100%'):
            start, end, _, recording, gesture = row
            gesture_df = param_df[param_df['recording'] == recording].iloc[start:end]
            if not gesture_df.empty:
                X.append(gesture_df[fields].to_numpy())
                y.append(gesture)
                
        return X, y