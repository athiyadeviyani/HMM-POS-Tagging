import inspect, sys, hashlib

# Hack around a warning message deep inside scikit learn, loaded by nltk :-(
#  Modelled on https://stackoverflow.com/a/25067818
import warnings
with warnings.catch_warnings(record=True) as w:
    save_filters=warnings.filters
    warnings.resetwarnings()
    warnings.simplefilter('ignore')
    import nltk
    warnings.filters=save_filters
try:
    nltk
except NameError:
    # didn't load, produce the warning
    import nltk

from nltk.corpus import brown
from nltk.tag import map_tag, tagset_mapping

# MY IMPORTS
from nltk.tag.hmm import HiddenMarkovModelTagger
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, LidstoneProbDist
from math import log
import numpy as np

if map_tag('brown', 'universal', 'NR-TL') != 'NOUN':
    # Out-of-date tagset, we add a few that we need
    tm=tagset_mapping('en-brown','universal')
    tm['NR-TL']=tm['NR-TL-HL']='NOUN'

class HMM:
    def __init__(self, train_data, test_data):
        """
        Initialise a new instance of the HMM.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :param test_data: the test/evaluation dataset, a list of sentence with tags
        :type test_data: list(list(tuple(str,str)))
        """
        self.train_data = train_data
        self.test_data = test_data

        # Emission and transition probability distributions
        self.emission_PD = None
        self.transition_PD = None
        self.states = []

        self.viterbi = []
        self.backpointer = []

    # Compute emission model using ConditionalProbDist with a LidstoneProbDist estimator.
    #   To achieve the latter, pass a function
    #    as the probdist_factory argument to ConditionalProbDist.
    #   This function should take 3 arguments
    #    and return a LidstoneProbDist initialised with +0.01 as gamma and an extra bin.
    #   See the documentation/help for ConditionalProbDist to see what arguments the
    #    probdist_factory function is called with.
    def emission_model(self, train_data):
        """
        Compute an emission model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The emission probability distribution and a list of the states
        :rtype: Tuple[ConditionalProbDist, list(str)]
        """
        # Prepare the data

        # Don't forget to lowercase the observation otherwise it mismatches the test data
        # Do NOT add <s> or </s> to the input sentences
        data = [(tag, word.lower()) for pairs in train_data for (word, tag) in pairs]

        # Compute the emission model
        emission_FD = ConditionalFreqDist(data)
        lidstone_estimator = lambda fd: LidstoneProbDist(fd, 0.01, fd.B() + 1)
        self.emission_PD = ConditionalProbDist(emission_FD, lidstone_estimator)
        
        for tag, word in data:
            if tag not in self.states:
                self.states.append(tag)

        return self.emission_PD, self.states

    # Access function for testing the emission model
    # For example model.elprob('VERB','is') might be -1.4
    def elprob(self,state,word):
        """
        The log of the estimated probability of emitting a word from a state

        :param state: the state name
        :type state: str
        :param word: the word
        :type word: str
        :return: log base 2 of the estimated emission probability
        :rtype: float
        """
        # IF block to catch if the model is not trained yet
        if self.emission_PD == None:
            self.emission_model(self.train_data)

        # Estimated log base 2 emission probability of emitting a word 'word' from a state 'state'
        return self.emission_PD[state].logprob(word) 

    # Compute transition model using ConditionalProbDist with a LidstonelprobDist estimator.
    # See comments for emission_model above for details on the estimator.
    def transition_model(self, train_data):
        """
        Compute an transition model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The transition probability distribution
        :rtype: ConditionalProbDist
        """
        # Prepare the data
        data = []
        tags = []

        # The data object should be an array of tuples of conditions and observations,
        # in our case the tuples will be of the form (tag_(i),tag_(i+1)).
        # DON'T FORGET TO ADD THE START SYMBOL </s> and the END SYMBOL </s>
        for s in train_data:
            start = ["<s>"]
            start.extend([tag for (word, tag) in s])
            start.extend(["</s>"])
            tags.extend(start)

        for i in range(len(tags) - 1):
            data.append((tags[i], tags[i+1]))

        # Compute the transition model
        transition_FD = ConditionalFreqDist(data)
        lidstone_estimator = lambda fd: LidstoneProbDist(fd, 0.01, fd.B() + 1)
        self.transition_PD = ConditionalProbDist(transition_FD, lidstone_estimator)

        return self.transition_PD

    # Access function for testing the transition model
    # For example model.tlprob('VERB','VERB') might be -2.4
    def tlprob(self,state1,state2):
        """
        The log of the estimated probability of a transition from one state to another

        :param state1: the first state name
        :type state1: str
        :param state2: the second state name
        :type state2: str
        :return: log base 2 of the estimated transition probability
        :rtype: float
        """
        # IF block to catch if the model is not trained yet
        if self.transition_PD == None:
            self.transition_model(self.train_data)

        # Estimated log base 2 transition probability from 'state1' to 'state2'
        return self.transition_PD[state1].logprob(state2) 

# Train the HMM
    def train(self):
        """
        Trains the HMM from the training data
        """
        self.emission_model(self.train_data)
        self.transition_model(self.train_data)

    # Part B: Implementing the Viterbi algorithm.

    # Initialise data structures for tagging a new sentence.
    # Describe the data structures with comments.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD
    # Input: first word in the sentence to tag
    def initialise(self, observation):
        """
        Initialise data structures for tagging a new sentence.
        :param observation: the first word in the sentence to tag
        :type observation: str
        """
        # Initialise step 0 of viterbi, including
        #  transition from <s> to observation
        # use costs (-log-base-2 probabilities)
        
        # The Viterbi data structure is a N by T table
        #   where T = number of observations
        #         N = number of states/tags
        # Each cell of the table contains the most probable path 
        # i.e. the maximum of all possible previous state sequences to arrive at that state
        self.viterbi = []

        # Initialise step 0 of backpointer

        # The backpointer data structure keeps track of the indexes which point to states with
        #   the highest viterbi probability (from the viterbi table)
        # This will used to backtrack from the last state to determine the final maximum path back 
        #   to the starting point
        # The backpointer data structure is a N by T table
        #   where T = number of observations
        #         N = number of states
        # Each cell of the table contains the state that has the maximum viterbi probability
        # in the previous observation
        self.backpointer = []

        # Go through the states
        # Calculate the probability of being at that state
        #   where probability = previous viterbi probability (at initialisation, this is none) + 
        #       probabiity to transition from previous state (at initialisation, the previous state is <s>) +
        #       emission probability of current state
        for state in self.states:
            
            # Transition probability is the probability to transition from state '<s>' to state 'state'
            transition_prob = -self.tlprob('<s>',state)
            # transition_prob = -self.transition_PD['<s>'].logprob(state)

            # Emission probability is the probability of observation 'observation' to current state 'state'
            emission_prob = -self.elprob(state, observation)
            # emission_prob = -self.emission_PD[state].logprob(observation)

            # Current probability = prev + transition_prob + emission_prob
            #       in this case, prev = 0
            current_state_probability = transition_prob + emission_prob
            self.viterbi.append([current_state_probability])
            
            # Backpointer initially points to the the start state
            self.backpointer.append([0])

    # Tag a new sentence using the trained model and already initialised data structures.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD.
    # Update the self.viterbi and self.backpointer datastructures.
    # Describe your implementation with comments.
    def tag(self, observations):
        """
        Tag a new sentence using the trained model and already initialised data structures.
        :param observations: List of words (a sentence) to be tagged
        :type observations: list(str)
        :return: List of tags corresponding to each word of the input
        """
        tags = []

        for t in range(1, len(observations)): # iterate over steps
            for s in range(len(self.states)): # iterate over states
                # Update the viterbi and backpointer data structures
                # Use costs, not probabilities
                state_probs = []

                # Go through all previous probabilities
                #   Calculate the probability of being in that state
                #       where probability = maximum of previous viterbi probabilities * 
                #               probability to transition from that previous state to the current state
                for prev_state in range(len(self.states)):

                    # Previous probability is the viterbi for the previous observation (at time step t-1)
                    previous_prob = self.viterbi[prev_state][t-1]

                    # Transition probability is the probability of transitioning from state prev_state to state s
                    transition_prob = -self.tlprob(self.states[prev_state], self.states[s])
                    #transition_prob = -self.transition_PD[self.states[prev_state]].logprob(self.states[s])

                    # Emission probability is the probability of observation t given current state s
                    emission_prob = -self.elprob(self.states[s], observations[t])
                    # emission_prob = -self.emission_PD[self.states[s]].logprob(observations[t])

                    # Current probability = previous_prob + emission_prob + transition_prob
                    #   We are summing them up because we have used the negative log probabilities
                    current_prob = previous_prob + emission_prob + transition_prob
                    state_probs.append(current_prob)

                # Add the maximum probability of all calculated probabilities to the respective path
                # for state s in the Viterbi table
                # Since we have used negative log probabilities, find the min
                max_prob = min(state_probs)
                self.viterbi[s].append(max_prob)

                # Add the index of the state which produced the maximum probability of all calculated
                # probabilities to the backpointer table for state s
                state_max_index = np.argmin(state_probs)
                self.backpointer[s].append(state_max_index)

        # Add a termination step with cost based solely on cost of transition to </s> , end of sentence.
        termination_probs = []

        # Go through all the states
        #   Add the transition probability from that state to the end state </s>
        for s in range(len(self.states)):
            prev_prob = self.viterbi[s][len(observations) - 1]
            transition_prob = -self.tlprob(self.states[s], ('</s>'))
            current_prob = prev_prob + transition_prob
            termination_probs.append(current_prob)

        # Reconstruct the tag sequence using the backpointer list.
        # Return the tag sequence corresponding to the best path as a list.
        # The order should match that of the words in the sentence.

        # First backpointer points to the index of the state which has the largest
        # probability in termination_probs
        #   i.e. maximum prev_prob + transition_prob 
        #           where transition_prob is the transition to </s> from current state
        last = np.argmin(termination_probs)

        # Append the corresponding state to the list of tags
        tags.append(self.states[last])

        # Iterate through the observation indexes from end to start
        #   Append the corresponding backpointed state to the list of tags
        #   Update the backpointer to point to the previous backpointed state
        for t in range(len(observations)-1, 0, -1):
            backpointer = self.backpointer[last][t]
            tags.append(self.states[backpointer])
            last = backpointer

        # We want to return the tags in the same order of the observations
        # Therefore, we have to reverse it
        tags.reverse()

        return tags

    # Access function for testing the viterbi data structure
    # For example model.get_viterbi_value('VERB',2) might be 6.42 
    def get_viterbi_value(self, state, step):
        """
        Return the current value from self.viterbi for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last step
        :type step: int
        :return: The value (a cost) for state as of step
        :rtype: float
        """
        # Viterbi data structure is T x N table
        #   where T is the number of observations (steps)
        #         N is the number of states
        return self.viterbi[self.states.index(state)][step]


    # Access function for testing the backpointer data structure
    # For example model.get_backpointer_value('VERB',2) might be 'NOUN'
    def get_backpointer_value(self, state, step):
        """
        Return the current backpointer from self.backpointer for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last step
        :type step: int
        :return: The state name to go back to at step-1
        :rtype: str
        """
        # First + if negative then count backwards from the end
        if step == 0 or step == -len(self.viterbi): 
            return '<s>'
        # Last + if negative then count backwards from the end
        if state == '</s>' and (step == len(self.viterbi) - 1 or step == -1): 
            return self.states[self.backpointer[0][steps]]
        return self.states[self.backpointer[self.states.index(state)][step]]

def answer_question4b():
    """
    Report a hand-chosen tagged sequence that is incorrect, correct it
    and discuss
    :rtype: list(tuple(str,str)), list(tuple(str,str)), str
    :return: your answer [max 280 chars]
    """

    # One sentence, i.e. a list of word/tag pairs, in two versions
    #  1) As tagged by your HMM
    #  2) With wrong tags corrected by hand
    tagged_sequence = [('``', '.'), ('My', 'DET'), ('taste', 'NOUN'), ('is', 'VERB'), ('gaudy', 'ADV'), ('.', '.')]
    correct_sequence = [('``', '.'), ('My', 'DET'), ('taste', 'NOUN'), ('is', 'VERB'), ('gaudy', 'ADJ'), ('.', '.')]
    # Why do you think the tagger tagged this example incorrectly?
    
    # QUANTITATIVE ANALYSIS
    # gaudy ADV -18.15 VERB ADV -3.81
    # gaudy ADJ -19.18 VERB ADJ -4.35

    answer = inspect.cleandoc("""\
    The HMM can only see 2-word histories, thus it can't see that the word 'gaudy' is actually an ADJ that describes the word 'taste'. The model tagged 'gaudy' as an ADV because 'gaudy' has a higher probability as an ADV and a VERB has a higher probability to be followed by an ADV.""")[0:280]

    return tagged_sequence, correct_sequence, answer

def answer_question5():
    """
    Suppose you have a hand-crafted grammar that has 100% coverage on
        constructions but less than 100% lexical coverage.
        How could you use a POS tagger to ensure that the grammar
        produces a parse for any well-formed sentence,
        even when it doesn't recognise the words within that sentence?

    :rtype: str
    :return: your answer [max 500 chars]
    """

    return inspect.cleandoc("""\
    To tag a sentence, we would need the transition and emission probabilities. For an unrecognised word, the emission probability will always be 0. However, if we make use of smoothing (e.g. by using the Lidstone estimator), it will address the problem by stealing probability mass from seen events and reallocating it to unseen events. Then, you will have a predicted tag assigned to the unrecognised word. The parsing algorithm should now be able to produce a parse for the well-formed sentence.""")[0:500]

def answer_question6():
    """
    Why else, besides the speedup already mentioned above, do you think we
    converted the original Brown Corpus tagset to the Universal tagset?
    What do you predict would happen if we hadn't done that?  Why?

    :rtype: str
    :return: your answer [max 500 chars]
    """

    return inspect.cleandoc("""\
    We converted the original Brown Corpus tagset to the Universal tagset because the Universal tagset has fewer tags. Having more tags means we're likely to have sparser data, i.e. the same word will have more different tags, thus that each (word, tag) pair will have less observations. This may cause lower confidence level on the probability model and accuracy on tag set will be much lower. The Universal tagset is more language agnostic (more generic) and will work better on other languages.""")[0:500]

# Useful for testing
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # http://stackoverflow.com/a/33024979
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def answers():
    global tagged_sentences_universal, test_data_universal, \
           train_data_universal, model, test_size, train_size, ttags, \
           correct, incorrect, accuracy, \
           good_tags, bad_tags, answer4b, answer5
    
    # Load the Brown corpus with the Universal tag set.
    tagged_sentences_universal = brown.tagged_sents(categories='news', tagset='universal')

    # Divide corpus into train and test data.
    # Test set = last 500 sentences
    # Train set = first (len(tagged_sentences_universal) - test_size) sentences
    # This split corresponds roughly to a 90/10 division
    test_size  = 500
    train_size = len(tagged_sentences_universal) - test_size
    test_data_universal = tagged_sentences_universal[-test_size:]
    train_data_universal = tagged_sentences_universal[:train_size]

    if hashlib.md5(''.join(map(lambda x:x[0],train_data_universal[0]+train_data_universal[-1]+test_data_universal[0]+test_data_universal[-1])).encode('utf-8')).hexdigest()!='164179b8e679e96b2d7ff7d360b75735':
        print('!!!test/train split (%s/%s) incorrect, most of your answers will be wrong hereafter!!!'%(len(train_data_universal),len(test_data_universal)),file=sys.stderr)

    # Create instance of HMM class and initialise the training and test sets.
    model = HMM(train_data_universal, test_data_universal)

    # Train the HMM.
    model.train()

    

    # Some preliminary sanity checks
    # Use these as a model for other checks
    e_sample=model.elprob('VERB','is')
    if not (type(e_sample)==float and e_sample<=0.0):
        print('elprob value (%s) must be a log probability'%e_sample,file=sys.stderr)

    t_sample=model.tlprob('VERB','VERB')
    if not (type(t_sample)==float and t_sample<=0.0):
           print('tlprob value (%s) must be a log probability'%t_sample,file=sys.stderr)

    if not (type(model.states)==list and \
            len(model.states)>0 and \
            type(model.states[0])==str):
        print('model.states value (%s) must be a non-empty list of strings'%model.states,file=sys.stderr)

    print('states: %s\n'%model.states)

    ######
    # Try the model, and test its accuracy [won't do anything useful
    #  until you've filled in the tag method
    ######
    s='the cat in the hat came back'.split()
    model.initialise(s[0])
    ttags = model.tag(s)

    print("Tagged a trial sentence:\n  %s"%list(zip(s,ttags)))

    v_sample=model.get_viterbi_value('VERB',5)
    if not (type(v_sample)==float and 0.0<=v_sample):
           print('viterbi value (%s) must be a cost'%v_sample,file=sys.stderr)

    b_sample=model.get_backpointer_value('VERB',5)
    if not (type(b_sample)==str and b_sample in model.states):
           print('backpointer value (%s) must be a state name'%b_sample,file=sys.stderr)





    # check the model's accuracy (% correct) using the test set
    correct = 0
    incorrect = 0

    incorrect_sentences = []
    i_s = []

    for sentence in test_data_universal:
        s = [word.lower() for (word, tag) in sentence]
        model.initialise(s[0])
        tags = model.tag(s)

        incorrect_bool = False

        for ((word,gold),tag) in zip(sentence,tags):
            if tag == gold:
                correct += 1
            else:
                incorrect_bool = incorrect_bool or True
                incorrect += 1
                
        if incorrect_bool:
            i_s.append(zip(sentence,tags))
            incorrect_sentences.append(zip(sentence,tags))
    
    for i in range(10):
        print("================= SENTENCE NUMBER: " + str(i + 1))
        print(list(i_s[i]))
        # print(' '.join([a for a, b in i_s[i]]))
        for ((word,gold),tag) in incorrect_sentences[i]:
            if tag != gold:
                print("Mistakenly tagged word: " + word)
                print("-- Expected tag: " + gold)
                print("-- Actual tag: " + tag)


    ### QUESTION 4b analysis
    # gaudy ADV -18.15 VERB ADV -3.81
    # gaudy ADJ -19.18 VERB ADJ -4.35

    print("================= QUESTION 4B ANALYSIS =================")

    print("COST OF GAUDY BEING ADV and ADJ")
    print(model.elprob('ADV','gaudy')) # -18.15227989016495     HIGHER 
    print(model.elprob('ADJ','gaudy')) # -19.178518971478535

    print("COST OF VERB FOLLOWED BY ADV and ADJ")
    print(model.tlprob('VERB','ADV')) # -3.812200102417784      HIGHER
    print(model.tlprob('VERB','ADJ')) # -4.349747810988496

    print("================= ACCURACY =================")

    accuracy = correct / (correct + incorrect)
    print('Tagging accuracy for test set of %s sentences: %.4f'%(test_size,accuracy))

    # Print answers for 4b, 5 and 6
    bad_tags, good_tags, answer4b = answer_question4b()
    print('\nA tagged-by-your-model version of a sentence:')
    print(bad_tags)
    print('The tagged version of this sentence from the corpus:')
    print(good_tags)
    print('\nDiscussion of the difference:')
    print(answer4b[:280])
    answer5=answer_question5()
    print('\nFor Q5:')
    print(answer5[:500])
    answer6=answer_question6()
    print('\nFor Q6:')
    print(answer6[:500])

if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1] == '--answers':
        import adrive2_embed
        from autodrive_embed import run, carefulBind
        with open("userErrs.txt","w") as errlog:
            run(globals(),answers,adrive2_embed.a2answers,errlog)
    else:
        answers()
