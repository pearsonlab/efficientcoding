# Efficient coding, channel capacity, and the emergence of retinal mosaics 

This code is supplement to the NeurIPS 2022 paper "Efficient coding, channel capacity, and the emergence of retinal mosaics" [1].

## Usage

This codebase requires the following Python packages: PyTorch 1.9, tensorboard, fire, tqdm, numpy, scipy, PIL, and matplotlib.

`train.py` is the main script, the following command will launch the training with the default arguments:

    python train.py

To train an image-only model based on the [Kyoto Natural Image Dataset](https://github.com/eizaburo-doi/kyoto_natim) as in [2]:

    python --data kyoto --frames 1 --temporal_kernel_size 1 --shape None

To list all options, run:

    python train.py --help

## Data

Because of the file size limit, the repo contains two small .npy files that contains a truncated video segment from one video from the dataset, which allows the training script to run but will not replicate the results in the paper. For the full replication, all videos except `butterfly*` from the [Chicago Motion Database](https://cmd.rcc.uchicago.edu/) need to be downloaded, converted to 512x512, 30fps videos, and extracted into numpy files of dtype uint8 that contain the luminocity channel in shape `(frames, height, width)`. The butterfly videos were excluded because they were out of the natural video distribution due to the insect screen (static & spatially narrow-band) and made optimization unstable. The per-video mean and standard deviation of pixels are precomputed in `stats.json` which is required for training.

## Files

- `data.py`: contains classes used for data retrieval. `VideoDataset` is for the natural video dataset (Fig 3), `FilteredVideoDataset` is for runnning the phase transition experiments for Fig 5, and `MultivariateGaussianDataset` is for generating multivariate Gaussian video segments using any covariance matrix. There are other data classes that we didn’t use for this particular research.

- `model.py`: `DiffExponentialShape` class is for the difference-of-exponential temporal RF parameterization. `Encoder` class implements the spatial linear filter, nonlinearity, and temporal convolutions, as well as ingredients to compute mutual information and firing rate constraint. `OutputMetrics` and `OutputTerms` classes combine the output values from the `Encoder` model, and `RetinaVAE` receives these metrics and then returns the objective value for the training loop.

- `shapes.py`: includes various spatial kernel shape classes, but we only used `DifferenceOfGaussianShape` class for this particular research.

- `temporal.py`: includes representative precomputed temporal kernel values to run the experiments in Supplementary Figure 3. 

- `util.py`: includes utility methods such as tools to draw plots on tensorboard

- `train.py`: the entrypoint of this project. It parses command line arguments and contains the main training loop. 


## Reference

> [1] Jun, Na Young, Greg D. Field, and John Pearson. "Efficient coding, channel capacity, and the emergence of retinal mosaics." *Advances in Neural Information Processing Systems* 35 (2022)

> [2] Jun, Na Young, Greg D. Field, and John Pearson. "Scene statistics and noise determine the relative arrangement of receptive field mosaics." Proceedings of the National Academy of Sciences 118.39 (2021)

For BibTeX:

```bibtex
@inproceedings{jun2022efficient,
    author = {Jun, Na Young and Field, Greg D. and Pearson, John M.},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {Efficient coding, channel capacity, and the emergence of retinal mosaics},
    volume = {35},
    year = {2022}
}

@article{jun2021mosaic,
    title={Scene statistics and noise determine the relative arrangement of receptive field mosaics},
    author={Jun, Na Young and Field, Greg D and Pearson, John},
    journal={Proceedings of the National Academy of Sciences},
    volume={118},
    number={39},
    year={2021},
    publisher={National Acad Sciences}
}
```
