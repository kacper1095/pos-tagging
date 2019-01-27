# Pos Tagging

## Description
This project includes all the code used during a private kaggle competition. This solution got 1st place on both public and private leaderboard.

## Requirements

- python 3.6
- pytorch 1.0
- tensorboardX

## Best solution description

1. Words embedded using fastText model available at Clarin-PL webpage ([link](https://nextcloud.clarin-pl.eu/index.php/s/luubhnS0AvjmtQc/download?path=%2F&files=kgr10.plain.skipgram.dim300.neg10.bin) to model).
2. Model uses following architecture:

It consists convolutional layers only.
At each level, 2 - 4 convolutional layers are used with kernel size 3 and different levels of kernel dilation.
Results of each of parallel layers are combined by weighted sum.
Weighting system is inspired by memory read weighting used in [NTM paper](https://arxiv.org/pdf/1410.5401.pdf).

## Additional info

For additional information / used data / trained models - send me a private message or an e-mail at kacper1095@gmail.com.

