from model_builder import *
import numpy as np
import sys
import itertools
import os

def calculate_perplexity(n : int, use_smoothing : bool,  test_data_file : str) -> None:
    '''Given a file path to a test corpus and a model to be evaluated, prints the model's perplexity'''
    if use_smoothing:
        print(f"Language Model: <N-gram with n = {n}. Uses Good-Turing Smoothing with a log-function>")
    else:
        print(f"Language Model: <N-gram with n = {n}. No smoothing used>")


    model = build_n_gram(n=n, use_smoothing=use_smoothing,save=False,build_corpus=False)

    with open(test_data_file, 'r') as file:
        test_corpus = file.read()
        file.close()

    # Clean the test corpus
    cleaned_test_corpus = clean_corpus(test_corpus)

    
    prob = get_model_probability(model=model, sentence=cleaned_test_corpus)
    if prob == 0:
        print("Perplexity = <Undefined>")
    else:
        print(f"Perplexity = <{prob ** (-1/len(cleaned_test_corpus.split()))}>") 



# calculates the perplexity of a given sentence 
def calculate_perplexity_sentence(n: int, use_smoothing: bool, sentence: str, clean_sentence:bool):
    '''Given a file path to a test corpus and a model to be evaluated, prints the model's perplexity'''

    # define Model Type
    model_type = ""
    if (n == 1):
        model_type = 'unigram'
    elif (n == 2):
        model_type = 'bigram'
    elif (n == 3):
        model_type = 'trigram'

    if use_smoothing:
        model_type += '_smoothed.data'
    else:
        model_type += '_no_smoothing.data'

    # Check if the model has been trained already and output file if so
    if os.path.exists(model_type):
        model = load_model(model_type)
    else:
        model = build_n_gram(n=n, use_smoothing=use_smoothing, save=False, build_corpus=False)

    # Clean the test corpus
    if not clean_sentence:
        cleaned_test_corpus= clean_corpus(sentence)
    else:
        cleaned_test_corpus = '+ '+sentence
    prob = get_model_probability(model=model, sentence=cleaned_test_corpus)
    if prob == 0:
        return None
    else:
        perplexity = prob**(-1/len(cleaned_test_corpus.split()))
        return perplexity


def unscramble(n:int,use_smoothing : bool,scrambled_file : str) -> None:
    '''Given a file path to a scrambled string, prints out the unscrambled string'''

    with open(scrambled_file, 'r') as file:
        scrambled_string = file.read()
        file.close()
    
    # Clean and prepare the input
    scrambled_string = re.sub(r'[^a-zA-Z\s]', '', scrambled_string).lower().strip() 
    words = scrambled_string.split() 
    
    # Morphological varients convert to standard form
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(w) for w in words]

    #define Model Type
    if(n==1):
        model_type='unigram'
    else:
        if(n==2):
            model_type='bigram'
        else:
            if(n==3):
                model_type='trigram'
            else:
                model_type='n'
    
    if (model_type!='n'):
        if use_smoothing:
            print(f"Language Model: <N-gram with n = {n} i.e. ",model_type,".Uses Good-Turing Smoothing with a log-function>")
            model_type+='_smoothed.data'
        else:
            print(f"Language Model: <N-gram with n = {n} i.e. ",model_type,".No smoothing used>")
            model_type+='_no_smoothing.data'
    else:
        if use_smoothing:
            print(f"Language Model: <N-gram with n = {n} .Uses Good-Turing Smoothing with a log-function>")
        else:
            print(f"Language Model: <N-gram with n = {n} . No smoothing used.")
            
    if model_type!='n':
        if os.path.exists(model_type):
            model=load_model(model_type)
    else:
        model = build_n_gram(n=n, use_smoothing=use_smoothing,save=False,build_corpus=False)
    
    original_prob = get_model_probability(model=model, sentence=scrambled_string)
    permutations = itertools.permutations(lemmatized_words)
    best_sequence, max_probability = None, float('-inf')
    
    #Find the sequence with maximum probability
    for permutation in permutations:
        sequence = ' '.join(permutation)
        probability = get_model_probability(model=model, sentence=sequence) 
        if probability > max_probability:
            max_probability = probability
            best_sequence = sequence 

    original_perplexity = 0
    unscrambled_perplexity = 0
    if original_prob == 0: 
        original_perplexity = "Undefined" 
    else:
        original_perplexity=(original_prob ** (-1/len(scrambled_string.split()))) 

    if max_probability == 0: 
        unscrambled_perplexity = "Undefined" 
    else:
        unscrambled_perplexity=(max_probability ** (-1/len(best_sequence.split())))
    
    print("Original Text: ",scrambled_string)
    print("Unscrambled sentence: ",best_sequence)
    print("Original Perplexity: ", original_perplexity)
    print("Unscrambled Perplexity: ",unscrambled_perplexity)
    

if __name__=="__main__":
    n_pp = None
    use_smoothing_pp = None
    n_unscramble = None
    use_smoothing_unscramble = None
    test_data_file = None
    scrambled_file = None
    for i in range (len(sys.argv)):
        if sys.argv[i] == '-model':
            n_pp = int(sys.argv[i + 1])
            if sys.argv[i + 2] in ['True']:
                use_smoothing_pp = True
            else:
                use_smoothing_pp = False
        if sys.argv[i] == '-evaluate':
            test_data_file = sys.argv[i + 1]
        if sys.argv[i] == '-unscramble':
            n_unscramble = int(sys.argv[i + 1])
            if sys.argv[i + 2] in ['True']:
                use_smoothing_unscramble = True
            else:
                use_smoothing_unscramble = False
            scrambled_file = sys.argv[i + 3]
    
    if n_pp and (use_smoothing_pp in [True, False]) and test_data_file:
        calculate_perplexity(n=n_pp, use_smoothing=use_smoothing_pp, test_data_file=test_data_file)

    if n_unscramble and (use_smoothing_unscramble in [True, False]) and scrambled_file:
        unscramble(n=n_unscramble, use_smoothing=use_smoothing_unscramble, scrambled_file=scrambled_file)
