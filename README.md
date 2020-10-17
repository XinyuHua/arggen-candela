## About

This repository contains code for our ACL19's paper [Argument Generation with Retrieval, Planning, and Realization](http://xinyuhua.github.io/resources/acl2019/acl2019.pdf). 

**Note**: Main modification to this repository includes using `torch.utils.Dataset` for data loading; tensorboard logging; and support for newer version of python and pytorch.

## Usage
### Requirement:

- python 3.7
- PyTorch 1.6.0
- numpy 1.15
- tensorboardX 2.1
- tqdm


### Data:

The dataset we used is currently held on Google drive, which can be accessed in this [link](https://drive.google.com/drive/folders/1fl9uxfkplJtbEppx4XeJ77nI0Iov_ZYL?usp=sharing).

### Pre-trained weights:

As described in the paper, we pre-train the encoder and realization decoder with extra data from changemyview. The pre-trained weights can be downloaded here: [encoder](https://drive.google.com/open?id=17dRozwLlWN_FgWQOyj-4fbsJC07bVZVx); [decoder](https://drive.google.com/open?id=1KO4FfxIQ1A8xKcT8QpTM6q28ZvLT_1Cd)

### To run

We assume the data to be loaded under `./data/` directory, and the pre-trained Glove embedding at `./embeddings/glove.6B.300d.txt`. The following snippet trains the model:

```shell script
python train.py \
    --exp-name=demo \
    --batch-size=16 \
    --max-epochs=30 \
    --save-freq=2 
```

Model checkpoints will be saved to `./checkpoints/[exp-name]/`, and tensorboard logs will be saved to `./runs/[exp-name]/`.

## Contact

Xinyu Hua (hua.x [at] northeastern.edu)

## License

See the [LICENSE](LICENSE) file for details.
