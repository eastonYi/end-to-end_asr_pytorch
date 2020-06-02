# Speech Transformer (Pytorch)
The implementation is based on [Speech Transformer: End-to-End ASR with Transformer]().
A PyTorch implementation of Speech Transformer [1], an end-to-end automatic speech recognition with [Transformer](https://arxiv.org/abs/1706.03762) network, which directly converts acoustic features to character sequence using a single nueral network.
This work is mainly done in Kuaishou as a . 

## Install
- Python3
- PyTorch 1.5
- [Kaldi](https://github.com/kaldi-asr/kaldi) (just for feature extraction)
- `pip install -r requirements.txt`
- `cd tools; make KALDI=/path/to/kaldi`

## Usage
### Quick start
```bash
$ cd egs/aishell
# Modify aishell data path to your path in the begining of run.sh
$ bash run.sh
```
That's all!

You can change parameter by `$ bash run.sh --parameter_name parameter_value`, egs, `$ bash run.sh --stage 3`. See parameter name in `egs/aishell/run.sh` before `. utils/parse_options.sh`.
### Workflow
- Data Preparation and Feature Generation
- Network Training
- Decoding
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
| SpeechTransformer | 12.8 | See egs/aishell/run.sh |

| SPKR   | #Snt | #Wrd | Corr  |  Sub  |  Del  | Ins  |  Err | S.Err |
| :---: | :-: | :----: |:----: |:----: |:----: |:----: |:----: |:----: |
| Sum/Avg| 7176 | 104765 | 89.6  |  9.9  |  0.4  |  0.3  | 10.7 |  57.0 |
## Reference
- [1] Yuanyuan Zhao, Jie Li, Xiaorui Wang, and Yan Li. "The SpeechTransformer for Large-scale Mandarin Chinese Speech Recognition." ICASSP 2019.
- [2] CIF
