### About the Project
The model can be trained on stories, books, articles, and the like, to learn to produce similar text. After training, the model can be given a sequence of words as its input to predict the next word. So, if the model is trained on stories, for example, you can ask the model to complete an incomplete story.

### Training
To train the model, simply run the following:
```
python 1-preprocess.py && python 2-word2vec.py && python 3-train.py
```

### Prediction
For prediction you will need:
1. The Trained model: `./results/model.h5`
2. Tokenizer: `./processed/tokenizer`
3. Sequences (to sample from): `./processed/sequences.txt`

### Configuration
Although not exhaustive, the configuration of the training can be found in `./helper/args.py`. Feel free to modify the model's architecture and functions at `./3-train.py`.

### Team
This work was part of our senior project in 2018 at KFUPM. Thanks to my colleagues Saleh Alresaini and Faris Alasmari, and also to our supervisor Dr. Lahouari Ghouti, for this great learning experience.
