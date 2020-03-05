import inspect, sys, hashlib
import nltk, inspect, math, numpy as np

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
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist, LidstoneProbDist
from math import log


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
        # raise NotImplementedError('HMM.emission_model')
        # TODO prepare data

        # Don't forget to lowercase the observation otherwise it mismatches the test data
        # Do NOT add <s> or </s> to the input sentences
        # [(tag, word.lower()) for i in a for (word, tag) in i]
        # data = [(tag, word.lower()) for (word, tag) in train_data]
        data = [(tag, word.lower()) for pairs in train_data for (word, tag) in pairs] 

        # TODO compute the emission model
        emission_FD = ConditionalFreqDist(data)
        self.emission_PD = ConditionalProbDist(emission_FD, LidstoneProbDist, 0.01)

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
        # raise NotImplementedError('HMM.elprob')
        test1 = log(self.emission_PD['NOUN'].prob('fulton')) # -7.47644570515
        test2 = log(self.emission_PD['X'].prob('fulton')) # -8.54286093816
        return test1, test2

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
        # raise NotImplementedError('HMM.transition_model')
        # TODO: prepare the data

        # The data object should be an array of tuples of conditions and observations,
        # in our case the tuples will be of the form (tag_(i),tag_(i+1)).
        # DON'T FORGET TO ADD THE START SYMBOL </s> and the END SYMBOL </s>
        tags = []

        # add start symbol and end symbol
        for tagged_sentence in train_data:
            i = ["<s>"]
            i.extend([tag for (word, tag) in tagged_sentence])
            i.extend(["</s>"])
            tags.extend(i)

        data = [(tags[i], tags[i+1]) for i in range(len(tags) - 1)]

        # TODO compute the transition model

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
        # raise NotImplementedError('HMM.tlprob')
        transition_PD = self.transition_model(self.train_data)
        return -transition_PD[state1].logprob(state2) # fixme

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
    # TODO: CHANGE THE COMMENTS!
    # Use the models stored in the variables: self.emission_PD and self.transition_PD
    # Input: first word in the sentence to tag
    def initialise(self, observation):
        """
        Initialise data structures for tagging a new sentence.

        :param observation: the first word in the sentence to tag
        :type observation: str
        """
        # raise NotImplementedError('HMM.initialise')
        # Initialise step 0 of viterbi, including
        #  transition from <s> to observation
        # use costs (-log-base-2 probabilities)

        # The Viterbi data structure contains the Viterbi path probabilities as a T by N
        # table where T is the number of observations and N is the number of states or tags.
        # Each cell contains the most probable path by taking the maximum over all possible
        # previous state sequences to arrive at that state.
        self.viterbi = []
        self.viterbi.append([])

        # Initialise step 0 of backpointer

        # The backpointer data structure keeps track of the best path of hidden states that
        # led to each state in a T by N table where T is the number of observations and N is
        # the number of states. Each cell contains the state which had the maximum viterbi
        # probability (from the Viterbi table) in the pervious time step (or observation).
        self.backpointer = []
        self.backpointer.append([])

        # At intialise, cost with +logprob or *prob
        for state in self.states:
            # Probability of transition from state '<s>' to state 'state'
            transition = self.transition_PD["<s>"].logprob(state)

            # Probability of observation 'observation' given the current state 'state'
            emission = self.emission_PD[state].logprob(observation)

            self.viterbi[0][state] = transition + emission
            self.backpointer[state] = [state]

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
        # raise NotImplementedError('HMM.tag')
        tags = []

        for t in range(1, len(observations)): # iterate over steps
            for s in range(len(self.states)): # iterate over states

                # list of probabilities and pointers to previous states
                # and take the most probable
                state_probabilities = []

                # go through all previous probabilities
                #   calculate the probability of being in that state
                #   probability = max of previous viterbi probabilities * 
                #       probability to transition from that previous state to current state
                for prevs in range(len(self.states)):

                    # previous prob is viterbu probability for the previous observation (t-1)
                    previous_prob = self.viterbi[prevs][t-1] # prev_prob is already neg log

                    # emission prob is probability of observation i given current state i
                    emission_prob = log(self.emission_PD[self.states[s]].prob(observations[t]), 2)

                    # transition prob is probability of transition from state prevs to state i
                    transition_prob = log(self.transition_PD[self.states[prevs]].prob(self.states[s]),2)

                    # current probability is the sum of all three above, negative log probabilities are
                    # used so we sum up instead of multiplying
                    probability     = previous_prob - emission_prob - transition_prob
                    state_probs     = state_probs + [probability]


                # add the max prob of calculated probabilities to the corresponding path in viterbi list
                self.viterbi[s].append(min(state_probs))

                # add the backpointer to the state which produced the
                # max prob of calculated probabilities
                self.backpointer[s].append(np.argmin(state_probs))


        
        # Add a termination step with cost based solely on cost of transition to </s> , end of sentence.
        end_probs = []

        # Similar to above, go through all states and add the probability for them to finish, that is, 
        # to transit from the state to end state </s>
        for i in range(len(self.states)):
            previous_prob = self.viterbi[i][len(observations) - 1]
            prob_of_end = log(self.transition_model[self.states[i]].prob("</s>"),2)
            total_prob = previous_prob - prob_of_end # negative log probability
            end_probs = end_probs + [total_prob]

        # Reconstruct the tag sequence using the backpointer list.
        # Return the tag sequence corresponding to the best path as a list.
        # The order should match that of the words in the sentence.

        # The first backpointer points to the state which had the largest probability
        # at the end (previous viterbi probability + probability to finish)
        last_bp = np.argmin(end_probs)

        # Append the last state to the tags list
        tags.append(self.states[last_bp])

        # Go through the observation indexes from end to start
        # Add the backpointed state to the tags list
        # Update the backpointer to the one which came from the previously backpointed state
        for i in range(len(observations)-1, 0, -1):
            bp = self.backpointer[last_bp][i]
            tags.append(self.states[bp])
            last_bp = bp

        # Reverse tags to match the order of observations
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
        :param step: The (0-origin) number of a step
        :type step: int
        :return: The value (a cost) for state as of step
        :rtype: float
        """
        raise NotImplementedError('HMM.get_viterbi_value')
        return ... # fix me

    # Access function for testing the backpointer data structure
    # For example model.get_backpointer_value('VERB',2) might be 'NOUN'
    def get_backpointer_value(self, state, step):
        """
        Return the current backpointer from self.backpointer for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step
        :type step: str
        :return: The state name to go back to at step-1
        :rtype: str
        """
        raise NotImplementedError('HMM.get_backpointer_value')
        return ... # fix me

def answer_question4b():
    """
    Report a hand-chosen tagged sequence that is incorrect, correct it
    and discuss
    :rtype: list(tuple(str,str)), list(tuple(str,str)), str
    :return: your answer [max 280 chars]
    """
    raise NotImplementedError('answer_question4b')

    # One sentence, i.e. a list of word/tag pairs, in two versions
    #  1) As tagged by your HMM
    #  2) With wrong tags corrected by hand
    tagged_sequence = 'fixme'
    correct_sequence = 'fixme'
    # Why do you think the tagger tagged this example incorrectly?
    answer =  inspect.cleandoc("""\
    fill me in""")[0:280]

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
    raise NotImplementedError('answer_question5')

    return inspect.cleandoc("""\
    fill me in""")[0:500]

def answer_question6():
    """
    Why else, besides the speedup already mentioned above, do you think we
    converted the original Brown Corpus tagset to the Universal tagset?
    What do you predict would happen if we hadn't done that?  Why?

    :rtype: str
    :return: your answer [max 500 chars]
    """
    raise NotImplementedError('answer_question6')

    return inspect.cleandoc("""\
    fill me in""")[0:500]

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
    test_size = 500
    train_size = 0 # fixme

    test_data_universal = tagged_sentences_universal[1:2] # fixme
    train_data_universal = tagged_sentences_universal[3:4] # fixme

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
    ttags = [] # fixme
    print("Tagged a trial sentence:\n  %s"%list(zip(s,ttags)))

    v_sample=model.get_viterbi_value('VERB',5)
    if not (type(v_sample)==float and 0.0<=v_sample):
           print('viterbi value (%s) must be a cost'%v_sample,file=sys.stderr)

    b_sample=model.get_backpointer_value('VERB',5)
    if not (type(b_sample)=='str' and b_sample in model.steps):
           print('backpointer value (%s) must be a state name'%b_sample,file=sys.stderr)


    # check the model's accuracy (% correct) using the test set
    correct = 0
    incorrect = 0

    for sentence in test_data_universal:
        s = [word.lower() for (word, tag) in sentence]
        model.initialise(s[0])
        tags = model.tag(s)

        for ((word,gold),tag) in zip(sentence,tags):
            if tag == gold:
                pass # fix me
            else:
                pass # fix me

    accuracy = 0.0 # fix me
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
