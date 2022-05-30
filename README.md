Implementation of "Over-the-Air Design of GAN Training for mmWave MIMO Channel Estimation"

Link to arXiv Preprint: https://arxiv.org/pdf/2205.12445

Code, results and data is structured with reference to the paper as given below:

/code - Contains /train to perform GAN and LOS Predictor trainings as described in Section IV and /test to perform Generative Channel Estimation as described in Section III. /baselines contains the OMP and EM-GM-AMP baselines implementation.

/results - Follows structure of Section VI-A through VI-D.

/data - Contains all the test data files, DFT matrices and (precoder, combiner, symbol) triplet used for Pilot GAN trainings. Download the training data from the URL given in /data/README.md

Code references:<br />
[1] https://github.com/eriklindernoren/PyTorch-GAN/ <br />
[2] https://github.com/nitinjmyers/Globecom2018_spatial_Zadoff_Chu/blob/master/OMPf.m <br />
[3] https://github.com/LeiDu-dev/FedSGD <br />
[4] https://sourceforge.net/projects/gampmatlab/
[5] https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
