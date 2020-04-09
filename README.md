# Informatics Undergraduate Honours Project

**Title**: _Automatic detection and classification of human head gestures_<br/>
**Supervisor**: [Dr. Hiroshi Shimodaira](http://homepages.inf.ed.ac.uk/hshimoda/)

<details>
<summary>
    <b>Click here to view the abstract.</b>
</summary>
<p>

> Head gestures are a simple, yet expressive form of human-to-human interaction, used as a medium for conveying ideas and emotions. Despite their simplicity as a form of communication, the accurate modelling and recognition of human head gestures has posed many challenges and provided many opportunities for machine learning research. The frequent use of motion-tracking devices, video, virtual-reality headsets and motion capture systems in the modern age of technology further motivates the need for effective head gesture recognition systems.
>
> In this dissertation, we focused primarily on the task of isolated head gesture recognition on rotation signals obtained from motion capture data. For this task, we performed in-depth research, application and evaluation of various widely-used sequence classification algorithms, including k-Nearest Neighbors with Dynamic Time Warping, Hidden Markov Models, Feed-Forward Neural Networks, and Recurrent Neural Networks with Long Short-Term Memory (LSTM). Comparisons between classifiers were done on the basis of recognition performance, which was measured with F1 score, and efficiency, which was measured in terms of peak memory consumption and fitting/prediction times.
>
> The most effective method of modelling gestures was a bidirectional multi-layer LSTM, which yielded an accuracy of **53.75±1%** and an F1 score of **52.30±1%**. This result is a vast improvement of **+15.1%** F1 score over previous works on the same dataset.

</p>
</details>

## Installation

This project requires the dependencies specified in `requirements.txt` (in addition to Python 3.7.4) in order for the notebooks to work.

It is suggested that you create a separate environment to contain the packages for the project.

```console
conda create -n inf4-hons python=3.7.4
conda activate inf4-hons
pip install -r requirements.txt
```

**Note**: You will not be able to re-run any of the notebooks as they rely on the proprietary `MoCap` dataset which I do not have permission to share.