# N-gram unscrambler by Ben Kurzion, Rohith Eshwarwak, and Akash Rana

## Corpus used
We did not want to pay for a full dataset from *English Corpora* but still wanted a corpus with an adequate number of tokens. 
To solve this issue, we combined the Corpus of Contemporary American English (COCA) and the Corpus of Historical American English (COHA) corpuses.
The corpuses were stemmed using the *nltk* library.
For model evaluation purposes, we split the combined corpus into 70% training data and 30% testing data.

You can find our corpus saved in **https://drive.google.com/drive/folders/1aqBkDyF4yoopot87uUuAhOzjXwvIcKv5?usp=sharing**

## Models implemented
We implemented a general *n-gram* model for any input *n*. We added functionality for both a smoothed and unsmoothed model. 

The smoothed model was smoothed using Good Turing smoothing. When using Good Turing smoothing, we adjust the *MLE* probabilities = (r + 1) * N<sub>(r + 1)</sub> / (N<sub>r</sub> * N)
- Where N<sub>r</sub> = the number of *n-grams* that appear *r* times
- And N = total number of *n_grams*

However, N<sub>r</sub> can be 0 in a finite dataset. We fit a log function to make sure that no N<sub>r</sub> is ever truly 0: log(N<sub>r</sub>) = a + b * log(r)

The unsmoothed model was implemented via standard MLE values for the parameters:

*n-gram* probability *MLE* = C(context, word) / C(context)
- Where C(*word_sequence*) = number of times *word_sequence* appears in the corpus

## Mapping models to files/lexicons
We have included the unigram lexicon counts in the *lexicon.txt* file. The file contains each unigram (unique, unstemmed words) with the corresponding number of times this word appears in the **full** corpus.
The lexicon is sorted by the frequency each word appears in descending order. It is formatted as *some_word : count*

The other files we have added are pre-trained models for unigram, bigram, and trigram. All models are uploaded with and without smoothing. The naming convention follows *modeltype_smoothed.data* or *modeltype_no_smoothing.data*

For example, a unigram model without smoothing would be saved as *unigram_no_smoothing.data*

The data stored in the model *.data* files is expressed as *sequence:probability*

We have a function *load_model()* that can be used to convert a model *.data* file into a python dictionary which maps a word sequence to its corresponding probability in the model.

## How to evaluate
Before evaluating our model, please navigate to our data and download the files *train_corpus.txt* and *test_corpus.txt*

These are saved in **https://drive.google.com/drive/folders/1aqBkDyF4yoopot87uUuAhOzjXwvIcKv5?usp=sharing**

To evaluate our models on their perplexity and their ability to unscramble strings, please refer to the *evaluate.py* file. This file can be run from the command line.
It takes a number of arguments in the form: 

*-model n use_smoothing -evaluate file_path*

Where 
- n : int
- use_smoothing : [True, False]
- file_path : str

*-unscramble n use_smoothing file_path*

Where 
- n : int
- use_smoothing : [True, False]
- file_path : str

If you want to check a model's perplexity using a text that you provide, you would use the first option. 
For example, if you want to evaluate the perplexity of a bigram with Good-Turing smoothing on a file called *test.txt*, you would run the file like so:

**-model 2 True -evaluate test.txt**

If you wanted to unscramble a string stored in a file, you would use the second option. 
For example, if you wanted to unscrable a string stored in a file called *scrambled.txt* using a unigram with Good-Turing smoothing, you would run the file like so:

**-unscramble 1 False scrambled.txt**

From the command line, you could run the *evaluate.py* file like so:

**python evaluate.py -model 2 True -evaluate test.txt -unscramble 1 False scrambled.txt**
