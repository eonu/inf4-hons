import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sequentia.preprocessing import downsample

# ggplot style
plt.style.use('ggplot')

def smart_downsample(X, m, method):
    X_new = []
    for x in X:
        T = len(x)
        n = int((T - T % -m) / m) - 1
        X_new.append(x if n in [0, 1] else downsample(x, n=n, method=method))
    return X_new

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
    
def show_class_counts(y, classes, rotate_xticks=False):
    """Display class counts for a dataset"""
    xs = [i for i in range(len(classes))]
    counts = [y.count(label) for label in classes]
    plt.figure(figsize=(8, 5))
    plt.title('Dataset class counts')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.bar(xs, counts, color='green')
    plt.xticks(xs, classes, rotation=90 if rotate_xticks else 0)
    plt.show()
    
def show_durations(X, bins=None):
    """Display a histogram of sequence durations"""
    durations = [len(x) for x in X]
    plt.figure(figsize=(8, 5))
    plt.title('Histogram of sequence durations')
    plt.xlabel('Duration (frames)')
    plt.ylabel('Count')
    plt.hist(durations, bins=bins)
    plt.show()
    
def show_accuracy_history(history):
    hist = history.history
    keys = hist.keys()
    plt.figure(figsize=(8, 5))
    plt.title('LSTM accuracy history during training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    if 'accuracy' in keys:
        plt.plot(hist['accuracy'], label='training')
    if 'val_accuracy' in keys:
        plt.plot(hist['val_accuracy'], label='validation')
    plt.legend()
    plt.show()
    
def show_loss_history(history):
    hist = history.history
    keys = hist.keys()
    plt.figure(figsize=(8, 5))
    plt.title('LSTM loss history during training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if 'loss' in keys:
        plt.plot(hist['loss'], label='training')
    if 'val_loss' in keys:
        plt.plot(hist['val_loss'], label='validation')
    plt.legend()
    plt.show()