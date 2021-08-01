# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
## Main idea
- Bidirectional Pre-training
- Fine-tuned 
- BERT uses masked language models

## Two step of the BERT
1. **pre-training**<br>: the model is trained on unlabeled data over different pre-training tasks<br>*Task #1*: Masked LM 사용 - 단어 몇 개를 [masked] 토큰으로 바꿔 original token 예측<br>*Task #2*: Next Sentence Prediction(NSP)
2. **fine-tuning**<br>: the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks

<img width="607" alt="스크린샷 2021-08-01 오후 7 03 44" src="https://user-images.githubusercontent.com/67621291/127767018-1623a96e-d706-4cdb-abeb-d6ea402dd6d6.png">


## Experiments
11 NLP tasks에 대해 실험 진행<br>
GLUE (The General Language Understanding Evaluation), SQuAD v1.1 (The Stanford Question Answering Dataset), SQuAD v2.0, SWAG (The Situation With Adversarial Generations) 사용
<br><br>


# Show and Tell: A Neural Image Caption Generator
## Main idea
<img width="500" alt="스크린샷 2021-08-01 오후 7 28 35" src="https://user-images.githubusercontent.com/67621291/127767646-c4051c33-fd40-4cb6-a687-8eac87822ec2.png"><br>
**Neural Image Caption (NIC)**<br>: Use CNN as an image "encoder", by first pre-training it for an image classification task and using the last hidden layer as an input to the RNN "decoder" that generates sentences.<br>
use LSTM as a sentence generator <br>
<img width="270" alt="스크린샷 2021-08-01 오후 7 36 28" src="https://user-images.githubusercontent.com/67621291/127767797-640f310a-3a66-4111-84e7-5d0088d0462f.png">
<br>

## Experiments
<img width="328" alt="스크린샷 2021-08-01 오후 8 31 41" src="https://user-images.githubusercontent.com/67621291/127769275-d1bb13f6-2344-4b97-a0e6-daab4256a480.png"><br>

**Word Embeddings**을 사용해 비슷한 단어 (horse, pony, donkey..)를 맞게 써서 더 많은 정보를 제공한다<br><br>

# Phrase-Based & Neural Unsupervised Machine Translation
## Main idea
1. **Initialization**: carefully initialize the MT system with an inferred bilingual dictionary
2. **Language Modeling**: leverage strong language models, via training the seq-to-seq system as a denoising autoencoder
3. **Iterative Back-translation**: turn the unsupervised problem into a supervised one by automatic generation of sentence pairs via back-translation
<img width="307" alt="스크린샷 2021-08-01 오후 10 40 04" src="https://user-images.githubusercontent.com/67621291/127772993-16dc80bd-5292-4372-97d1-7306a3552469.png">

위의 세 가지를 1. Unsupervised NMT, 2. Unsupervised PBSMT 에 대해 진행<br><br>

## Unsupervised NMT
### Initialization
1. join the monolingual corpora
2. apply BPE tokenization on the resulting corpus
3. learn token embeddings on the same corpus

### Language Modeling 
In NMT, language modeling is accomplished via denoising autoencoding, by minimizing:<br>
<img width="256" alt="스크린샷 2021-08-01 오후 10 48 31" src="https://user-images.githubusercontent.com/67621291/127773249-5dd09241-9bd3-4c37-8d9d-3353e133fa75.png">

### Back-translation
train two MT models by minimizing the loss:<br>
<img width="256" alt="스크린샷 2021-08-01 오후 10 52 06" src="https://user-images.githubusercontent.com/67621291/127773363-9c836857-2cbf-45b9-b8a1-be8960427ffa.png"><br>
when minimizing this objective function, we do not back-prop through the reverse model 

### Sharing Latenet Representations
To prevent the model from cheating by using different subspaces for the language modeling and translation tasks.<br>
In order to share the encoder representations, we share all encoder parameters across the two languages <br>
Similarly, we share the decoder parameters across the two languages. 
<br><br>

## Unsupervised PBSMT
PBSMT - Phrase-Based Statistical Machine Translation: perform well on low-resource language pairs<br>

### Initialization
populate the initial phrase tables <br>
Phrase tables are populated with the scores of the translation of a source word to:<br>
<img width="300" alt="스크린샷 2021-08-01 오후 11 04 07" src="https://user-images.githubusercontent.com/67621291/127773733-244e0e18-eecb-41bd-a5dd-539b352a6bcc.png">

### Language Modeling
Both in the source and target domains we learn smoothed n-gram language models using KenLM

### Iterative Back-translation
To jump-start the iterative process, we use the unsupervised phrase tables and the language model on the target side to construct a seed PBSMT.<br>
Then, use this model to translate the source monolingual corpus into the target language (back-translation step)<br>
Once the data has been generated, we train a PBSMT in supervised mode to map the generated data back to the original source sencence.<br>
We perform both generation and training process but in the reverse direction.<br>
Repeat these steps as many times as desired.<br><br>

<img width="304" alt="스크린샷 2021-08-01 오후 11 10 37" src="https://user-images.githubusercontent.com/67621291/127773954-9c1640b8-67e2-4418-984b-79c53adb9191.png">


## Experiments
<img width="616" alt="스크린샷 2021-08-01 오후 11 12 36" src="https://user-images.githubusercontent.com/67621291/127774019-9c62811c-d70d-4c34-b47c-3d5da0019621.png">
