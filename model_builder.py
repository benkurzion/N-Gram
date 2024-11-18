import os
import numpy as np
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import math

def read_all_text_files(directory_path) -> str:
    '''Reads all the text files in a given directory. Returns a single string with all the text appended onto one another'''
    corpus = ""
    
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r') as file:
                    corpus += file.read() + " "
                    file.close()
            except Exception as e:
                print(f"Bad read on file {filename}: {str(e)}")
    except Exception as e:
        print(f"Bad access on directory: {str(e)}")
        corpus = ""
    return corpus


    
def clean_corpus(corpus : str) -> str:
    '''Takes dirty data and removes non-alphabet/non-space characters. Stems every token in corpus. Returns cleaned corpus'''

    # Regex remove non-alphabet/non-space/non-period characters and all lower case 
    corpus = re.sub(r'[^a-zA-Z\s\.]', '', corpus)
    corpus = corpus.lower()

    # Replace all periods with '+' which indicates a start/end token
    corpus = re.sub(r'\.', ' +', corpus)
    corpus = re.sub(r'\n', ' ', corpus)

    cleaned_corpus = ""
    wnl = WordNetLemmatizer()

    words = corpus.split()
    for w in words:
        cleaned_corpus +=  str(wnl.lemmatize(w) + " ")

    # Add start and end token '+' if not already in string
    cleaned_corpus = cleaned_corpus.strip()
    if cleaned_corpus[0] != '+':
        cleaned_corpus = '+ ' + cleaned_corpus
    if cleaned_corpus[-1] != '+':
        cleaned_corpus = cleaned_corpus + ' +'

    return cleaned_corpus


    
def get_lexicon_counts(corpus : str) -> None:
    '''Saves the sorted counts for each unigram in a txt file.'''
    unigram_counts = {}
    words = corpus.split()
    for word in words:
        if word in unigram_counts:
            unigram_counts[word] += 1
        else:
            unigram_counts[word] = 1
    
    # Sort the dictionary by value
    unigram_counts = {k: v for k, v in sorted(unigram_counts.items(), key=lambda item: item[1], reverse=True)}
    with open("lexicon.txt", 'w') as file:
        for word, count in unigram_counts.items():
            file.write(f"{word} : {count}\n")
        file.close()



def build_n_gram(n=1, use_smoothing=True, file_name='ngram', save=True, build_corpus=True) -> dict:
    '''
    Trains a n-gram model and saves the model parameters in a .data file

    Parameters
    --------------

    n : int, default=1
        Controls the context window and number of parameters that the n-gram model uses. Context window is n-1

    use_smoothing: bool, default=True
        Flag that determines whether or not to use the Good-Turing Smoothing

    file_name : str, default='ngram'
        The file name which all model parameters will be saved. 
        Recommended format: 'bigram_smoothed' or 'unigram_no_smooth' etc.

    save : bool, default=True
        Determines whether or not to save the model as a .data file

    build_corpus : bool, default=True
        Determines whether to build the corpus from scratch or to load a corpus from memory

    :returns: Dictionary mapping the sequences and their corresponding probabilities
    '''
    if build_corpus:
        # Get the corpus
        COHA_FILEPATH = 'COHA Corpus'
        COCA_FILEPATH = 'COCA Corpus'
        coha_corpus = read_all_text_files(COHA_FILEPATH)
        coca_corpus = read_all_text_files(COCA_FILEPATH)

        # Clean the corpus
        coha_corpus = clean_corpus(coha_corpus)
        coca_corpus = clean_corpus(coca_corpus)

        # Combine the corpuses
        cleaned_corpus = coha_corpus + coca_corpus
        
        # Split the corpus into train test sets with 70/30
        train_ratio = int(len(cleaned_corpus.split()) * 0.7)
        train_corpus = ' '.join(cleaned_corpus.split()[:train_ratio])
        test_corpus = ' '.join(cleaned_corpus.split()[train_ratio:])

        with open("train_corpus.txt", 'w') as file:
            file.write(train_corpus)
            file.close()

        with open("test_corpus.txt", 'w') as file:
            file.write(test_corpus)
            file.close()
    else:    
        with open('train_corpus.txt', 'r') as file:
            train_corpus = file.read()
            file.close()
    
    
    words = train_corpus.split()
    sequence_counts = {}
    total_num_n_grams = 0

    # Sliding window through the entire corpus
    for i in range (max(0, n - 1), len (words)):
        sequence = words[i - (n - 1) : i + 1]
        # Convert the list sequence to a space-separated string of tokens
        sequence = ' '.join(sequence)
        total_num_n_grams += 1
        if sequence in sequence_counts: 
            sequence_counts[sequence] += 1
        else:
            sequence_counts[sequence] = 1

    
    if use_smoothing:
        ''' 
        n-gram probability MLE = C(context, word) / C(context)
            Where C(word_sequence) = number of times word_sequence appears in the corpus
        
        When using Good Turing smoothing, we adjust the MLE probabilities = (r + 1) * N_(r + 1) / (N_r * N)
            Where N_r = the number of n-grams that appear r times
            And N = total number of n_grams

        However, N_r can be 0 in a finite dataset. We will use further smoothing and fit a log function
            log(N_r) = a + b * log(r)
        '''
        # Calculate the frequencies using key value pair (r, N_r)
        frequencies = {}
        for sequence, count in sequence_counts.items():
            if count in frequencies:
                frequencies[count] += 1
            else:
                frequencies[count] = 1

        # Fit the log function to the key value frequence pairs
        X = list(frequencies.keys())
        y = list(frequencies.values())

        parameters = np.polyfit(np.log(X), np.log(y), 1)
        a = parameters[1]
        b = parameters[0]

        def get_smoothed_N_r(r : int ) -> float:
            return math.exp(a + b  * np.log(r))

        
        # Calculate the probabilities for each seen sequence
        probabilities = {}
        for sequence, count in sequence_counts.items():
            probabilities[sequence] = (count + 1) * get_smoothed_N_r(count + 1) / (get_smoothed_N_r(count) * total_num_n_grams)

        # Add the probability for unseen sequences
        # N_0 cannot be approximated by a log function and therefore needs to be explicitely calculated = number of unseen n-grams given a vocabulary V
        vocabulary_size = len(set(words))
        n_0 = (vocabulary_size ** n) - len(sequence_counts)
        if n_0 == 0:
            probabilities["UNSEEN"] = 0
        else:
            probabilities["UNSEEN"] = get_smoothed_N_r(1) / (n_0 * total_num_n_grams)
    else:
        if n > 1:
            context_counts = {}

            # Sliding window through the entire corpus to collect context counts
            for i in range (max(0, n - 1), len (words)):
                context = words[i - (n - 1) : i]
                context = ' '.join(context)
                if context in context_counts: 
                    context_counts[context] += 1
                else:
                    context_counts[context] = 1
            # Calculate the probabilities for each seen sequence
            probabilities = {}
            for sequence, count in sequence_counts.items():
                arr_seq = sequence.split()
                context = arr_seq[:(n - 1)]
                context =' '.join(context)
                probabilities[sequence] = count / context_counts[context]
        else:
            # Unigram special case where MLE = C(word) / (Number of unique unigrams)
            probabilities = {}
            for word, count in sequence_counts.items():
                probabilities[word] = count / len(sequence_counts)

    # Normalize the probabilities 
    total_probability = sum(probabilities.values())
    for sequence, probability in probabilities.items():
        probabilities[sequence] = probability / total_probability
        
    if save:   
        # Save the data
        file_name = file_name + '.data'
        with open(file_name, 'w') as file:
            for sequence, probability in probabilities.items():
                file.write(f"{sequence}:{probability}\n")
            file.close()

    return probabilities


def load_model(file_path=None) -> dict:
    '''
    Loads the n-gram parameters from a file. Expressed as a dictionary with key value pairs: (sequence, probability)

    Parameters
    -----------

    file_path : str, default=None
        The file path to the .data file where the model parameters are stored
    '''
    model = {}
    with open(file_path, 'r') as file:
        for line in file:
            split = line.split(':')
            sequence = split[0]
            probability = split[1]
            # Remove the \n
            probability = probability[:-1]
            model[sequence] = float(probability)
        file.close()
    return model





def get_model_probability(model : dict, sentence : str) -> float:
    '''Returns the probability the model will generate this sentence'''

    words = sentence.split()
    probability = 1.0
    n = len(list(model.keys())[0].split())

    # There are max(0, n-1) tokens at the beginning of the input sentence that do not have adequate context
    # The chain rule = p(w_1) * ... * p(w_(n-1)) * p(w_n | w_1, ..., w_(n-1))
    # We need to find the probability of p(w_1) * ... * p(w_(n-1))
    # Use a lower order model: p(w_1 | +) = bigram probability of the first word given start
    # p(w_2 | + , w_1) = trigram probability of the second word given the start string and the first word

    # Calculating an unsmoothed probability is a simple MLE: P(w | context) = c(context, w) / c(context)
    # However, if the model we are trying to calculate the probability for is smoothed, then we must also use the smoothed version of the lower order models
    # The only way to get a single probability of a smoothed model is to calculate the probabilities for every sequence and then smooth
    
    if n > 1 and "UNSEEN" in model.keys(): # Smoothed, higher order than unigram
        temp_use_smoothing = True
    elif n > 1: # Unsmoothed, higher order than unigram
        temp_use_smoothing = False
    
    # start from index 1 as index 0 has a '+' token
    temp_n = 2
    for i in range (1, n-1): 
        temp_model = build_n_gram(n=temp_n, use_smoothing=temp_use_smoothing, save=False, build_corpus=False)
        sequence = ' '.join(words[0 : i + 1])
        if sequence in temp_model:
            probability = probability * temp_model[sequence]
        elif "UNSEEN" in temp_model:
            probability = probability * temp_model["UNSEEN"]
        else:
            return 0
        temp_n += 1
    
    # Sliding window for chain rule calculation
    for i in range (max(0, n - 1), len (words)):
        sequence = words[i - (n - 1) : i + 1]
        # Convert the list sequence to a space-separated string of tokens
        sequence = ' '.join(sequence)
        if sequence in model:
            probability = probability * model[sequence]
        elif "UNSEEN" in model:
            probability = probability * model["UNSEEN"]
        else:
            return 0
    return probability
