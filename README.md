# Fast Self-attentive Multimodal Retrieval

**Note**: The original code for our paper "Fast Self-attentive Multimodal Retrieval" is protected. For providing a public version, we forked this code from: https://github.com/fartashf/vsepp/ and adapted it by adding the self-attentive mechanism along with the main proposed methods. 


## Dependencies
We recommended to use Anaconda for the following packages.

* Python 2.7
* [PyTorch](http://pytorch.org/) (>0.1.12)
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```

## Download data

Download the dataset files and pre-trained models. We use splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). To use full image encoders, download the images from their original sources [here](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html), [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).

```bash
wget http://lsa.pucrs.br/jonatas/seam-data/irv2_precomp.tar.gz
wget http://lsa.pucrs.br/jonatas/seam-data/resnet152_precomp.tar.gz
wget http://lsa.pucrs.br/jonatas/seam-data/vocab.tar.gz
```

We refer to the path of extracted files for `*_precomp.tar.gz` as `$DATA_PATH` and 
files for `models.tar.gz` (*models are coming up soon*) as `$RUN_PATH`. Extract `vocab.tar.gz` to `./vocab` 
directory.


## Training new models
Run `train.py`:

```bash
python train.py --data_path "$DATA_PATH" --data_name irv2_precomp --logger_name 
runs/seam-e/irv2_precomp/ 
```

Arguments used to train pre-trained models:

| Method    | Arguments |
| :-------: | :-------: |
| SEAM-E    | `--text_encoder seam-e --att_units 300 --att_hops 30 --att_coef 0.5 --measure order --use_abs` |
| SEAM-C    | `--text_encoder seam-c  --att_units 300 --att_hops 10 --att_coef 0.5 --measure order --use_abs` |
| SEAM-G    | `--text_encoder seam-g --att_units 300 --att_hops 30 --att_coef 0.5 --measure order --use_abs` |
| Order     | `--text_encoder gru  ` |

Available text encoders: 
* SEAM-E (`seam-e`): Self-attention directly over word-embeddings
* SEAM-C (`seam-c`): Self-attention over two parallel convolutional layers and over the word inputs. 
* SEAM-G (`seam-g`): GRU + Self-attention


Note that some default arguments in this repository are different from the original one:

`--learning_rate .001 --margin .05`


## Evaluate pre-trained models

```python
from vocab import Vocabulary
import evaluation
evaluation.evalrank("$RUN_PATH/model_best.pth.tar", data_path="$DATA_PATH", split="test")'
```

To do cross-validation on MSCOCO, pass `fold5=True` with a model trained using 
`--data_name coco`.


## Results

[**Coming up soon**] Results achieved using this repository (COCO-1cv test set): 

| Method    | Features | R@1 | R@10| R@1 | R@10 |
| :-------: | :----: | :-------: | :-------: | :-------: | :-------: |
| SEAM-E    | `resnet152_precomp` | |
| SEAM-C    | `resnet152_precomp` | |
| SEAM-G    | `resnet152_precomp` | |



## Reference

If you found this code useful, please cite the following papers:

    @article{wehrmann2018fast,
      title={Fast Self-Attentive Multimodal Retrieval},
      author={Wehrmann, Jônatas and Armani, Maurício and More, Martin D. and Barros, Rodrigo C.},
      journal={IEEE Winter Conf. on Applications of Computer Vision (WACV'18)},
      year={2018}
    }
    
    @article{faghri2017vse++,
      title={VSE++: Improved Visual-Semantic Embeddings},
      author={Faghri, Fartash and Fleet, David J and Kiros, Jamie Ryan and Fidler, Sanja},
      journal={arXiv preprint arXiv:1707.05612},
      year={2017}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)