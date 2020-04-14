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

## Installation and reproducing results

This project requires the dependencies specified in `requirements.txt` (in addition to Python 3.7.4) in order for the notebooks to work.

It is suggested that you create a separate environment to contain the packages for the project.

```console
conda create -n inf4-hons python=3.7.4
conda activate inf4-hons
pip install -r requirements.txt
```

**Note**: You will not be able to re-run any of the notebooks as they rely on the proprietary `MoCap` dataset which I do not have permission to share.

If you do have access to these files, the following files in the `notebooks/` directory contain the implementation necessary for the reproduction of results:

- `MoCap Recognition.ipynb`: Contains the code for training and evaluating the performance of the final classifiers (on the validation set).
- `utils.py`: Contains utility functions and classes for performance evaluation, storing results, loading gesture parameters and annotations, and visualizations.
- `nn.py`: Contains the implementation of a class used to create the Feed-Forward and Long Short-Term Memory neural networks.

Other sub-directories of importance within the `notebooks/` directory include:

- `Experiments/`: Contains results of the repeated experiments described in the dissertation.
- `Plot Scripts/`: Contains scripts for producing the plots used in the dissertation.
- `Plots/`: Contains the plots used in the dissertation.

## Contributions

- `notebooks/`: Main code for this dissertation.
- `sequentia/`: Version `0.7.0a1` copy of the [Sequentia](https://github.com/eonu/sequentia) library that was developed for this project.
- `params/organize.rb`: Code for sorting the gesture signal files into separate sub-directories for each speaker and file format.
- `params/rename.rb`: Code for renaming gesture signal files.
- `params/rov2csv.rb`: Code for converting the ROV gesture parameter data into CSV format.
- `params/Rakefile`: Commands for using `rov2csv.rb` to generate CSV files containing ROV data.
- `annotations/Rakefile`: Commands for using the `eaf2csv.py` script created by Roxana Novac, which converts gesture annotations from the XML-like EAF format used by [ELAN](https://archive.mpi.nl/tla/elan), to CSV which is more usable.

All other files except `README.md`, `requirements.txt` and Git-related files, were created by either Dr. Shimodaira or ex-MSc students.

---

<p align="center">
    This work was submitted in partial fulfillment of the requirements for the <a href="https://www.ed.ac.uk/studying/undergraduate/degrees/index.php?action=view&code=G400">BSc Computer Science</a> degree at the <a href="https://www.ed.ac.uk/informatics">School of Informatics, University of Edinburgh</a>.
</p>

<p align="center">
    All aforementioned contribution items are protected under the <a href="https://creativecommons.org/licenses/by/4.0/">Creative-Commons Attribution 4.0 International</a> license unless stated otherwise.
</p>

<p align="center">
    <em>© Edwin Onuonga, 2020</em>
</p>