# word-rnn-keras
In the previous expriment, we have implement char-rnn model which can be used to predict next char given a sequence of previous characters. 
This will then allow us to generate new text one character at a time.
In this expriment, we will implement word-rnn model which can be used to prdict next word given a sequence of previous words.
This will then allow us to generate a new sentence one word at a time.

The model implementatinons in localhost : 10.15.82.136  xucan/github/new_word_rnnlm/2.3_data/

Dataset: MovieTriples

Training set:   the length of sentences  is between 10 and 30. and all the words in the sentence should be aware in embdding dict.
                          We selected 9718 sentences from "Training_Shuffled_Dataset.txt", The training set will also be used as Validation set.

Dict:  The dict consists of all the words in the  training set which length is 4416. The first four words is {'fooo':0, '<unk>':1,'<s>':2, '</s>':3}
           'fooo':0 is convenient for Embedding layers and mask layer . '<s>': is the start word of a sentence . 
           '<unk>' : is the unknowen words . '</s>' is the last word which is convenient for label and stop in the process of generate sentence. 

Input:  X = 9718 * 31 (the max len and a start word '<s>') * 4416 (one-hot)
              Y = 9718 * 31
