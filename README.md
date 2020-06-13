# Speech Transformer (Pytorch)
The implementation is based on [Speech Transformer: End-to-End ASR with Transformer](https://github.com/kaituoxu/Speech-Transformer).
A PyTorch implementation of Speech Transformer network, which directly converts acoustic features to character sequence using a single nueral network.
This work is mainly done in Kuaishou as an intern.

## Install
- Python3
- PyTorch 1.5
- [Kaldi](https://github.com/kaldi-asr/kaldi) (just for feature extraction)
- `pip install -r requirements.txt`

## Usage
### Quick start
```bash
$ cd egs/aishell
# Modify aishell data path to your path in the begining of run.sh
$ bash transofrmer.sh
```
That's all!

You can change parameter by `$ bash transofrmer.sh --parameter_name parameter_value`, egs, `$ bash run.sh --stage 3`. See parameter name in `egs/aishell/run.sh` before `. utils/parse_options.sh`.

### Workflow
- Data Preparation and Feature Generation
    TODO: using the scripts in data_prepare
- Network Training

- Decoding
    change the transofrmer.sh
### More detail
`egs/aishell/run.sh` provide example usage.
```bash
# Set PATH and PYTHONPATH
$ cd egs/aishell/; . ./path.sh
# Train
$ train.py -h
# Decode
$ recognize.py -h
```

#### How to resume training?
```bash
$ bash run.sh --continue_from <model-path>
```

## Results
| Model | CER | Config |
| :---: | :-: | :----: |
| LSTMP | 9.85| 4x(1024-512). See [kaldi-ktnet1](https://github.com/kaituoxu/kaldi-ktnet1/blob/ktnet1/egs/aishell/s5/local/nnet1/run_4lstm.sh)|
| Listen, Attend and Spell | 13.2 | See [Listen-Attend-Spell](https://github.com/kaituoxu/Listen-Attend-Spell)'s egs/aishell/run.sh |
| SpeechTransformer | 10.7 | See egs/aishell/run.sh |

| Model | #Snt | #Wrd |   Sub  |  Del  | Ins  |  CER |
| :---: | :-: | :----: |:----: |:----: |:----: | :----: |
| SpeechTransformer |  7176 | 104765 | 9.9  |  0.4  |  0.3  | 10.7 |
| Conv_CTC_Transformer |  7176 | 104765 | |
| Conv_CTC |  7176 | 104765 | |
| CIF | 7176 | 104765 | 12.7 | 0.3 | 3.5 | 16.5 |

## Acknowledgement
- The framework and speech-transofrmer baseline is based on [Speech Transformer: End-to-End ASR with Transformer](https://github.com/kaituoxu/Speech-Transformer)
- `src/transformer/conv_encoder.py` refers to https://github.com/by2101/OpenASR.
- The core implement of CIF algorithm is checked by Linhao Dong (the origin author of CIF)


## Reference
- [1] Yuanyuan Zhao, Jie Li, Xiaorui Wang, and Yan Li. "The SpeechTransformer for Large-scale Mandarin Chinese Speech Recognition." ICASSP 2019.
- [2] L. Dong and B. Xu, “CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition Linhao,” in Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, 2017, vol. 2017-Augus, pp. 3822–3826.
