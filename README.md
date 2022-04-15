# DA-GPN

## Requirements
* torch >= 1.8.1

* transformers >= 3.4.0

* ujson

* tqdm
  

## TACRED Dataset
The TACRED dataset can be obtained from [this link](https://nlp.stanford.edu/projects/tacred/). The TACREV and Re-TACRED dataset can be obtained following the instructions in [Tacrev](https://github.com/DFKI-NLP/tacrev) and [Re-TACRED](https://github.com/gstoica27/Re-TACRED), respectively. The expected structure of files is:
```
DA-GPN
 |-- dataset
 |    |-- tacred
 |    |    |-- train.json        
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |    |-- dev_rev.json
 |    |    |-- test_rev.json
 |    |-- retacred
 |    |    |-- train.json        
 |    |    |-- dev.json
 |    |    |-- test.json
```

### Training and Evaluation
Train the DA-GPN model:

```bash
>> sh run_roberta_tacred.sh    # TACRED and TACREV
>> sh run_roberta_retacred.sh  # Re-TACRED
```
The results on TACRED and TACREV can be obtained in one run as they share the same training set.

## DialogRE Dataset 
This DialogRE dataset can be downloaded at: https://github.com/nlpdata/dialogre. You can download and unzip **BERT-base-uncased** from https://github.com/google-research/bert

```bash
>> sh run_dialog.sh    # Dialog
```
Note: We use Roberta large as the backbone of BERT module, and perform our experiments on GTX 3090 card.

## Related Repo

Codes are adapted from the repo of the 2021 paper [An Improved Baseline for Sentence-level Relation Extraction].

