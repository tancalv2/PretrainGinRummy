import numpy as np
from random import randint

from rlcard.core import Card, Card2
from rlcard.GinRummyUtil import GinRummyUtil

class SimpleGinRummyAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, action_num):
    #     ''' Initilize the random agent

    #     Args:
    #         action_num (int): The size of the ouput action space
    #     '''
        self.action_num = action_num
        self.use_raw = False

    def step(self, state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''

        # Plane (5x52)      Feature
        # 0  currHand       the cards in current player's hand
        # 1  topCard        the top card of the discard pile
        # 2  deadCard       the dead cards: cards in discard pile (excluding the top card)
        # 3  oppCard        opponent known cards: cards picked up from discard pile, but not discarded
        # 4  unknownCard    the unknown cards: cards in stockpile or in opponent hand (but not known)
        states = state['obs']
        currHandState = states[0,:]
        # print(state['legal_actions'])

        if 0 in state['legal_actions'] or 1 in state['legal_actions'] or 5 in state['legal_actions']:
            # Score player, or gin action
            # print('Action', state['legal_actions'])
            return np.random.choice(state['legal_actions'])
        else:
            currHandind = np.where(currHandState == 1)[0]
            currHand = []
            for i in currHandind:
                currHand.append(Card2(int(i%13), int(i//13)))

            # 1. Pickup card from discard pile if card can be melded with current hand
            if len(currHand) == 10:
                faceUpCardState = states[1,:]
                faceUpCardind = np.where(faceUpCardState == 1)[0]
                faceUpCard = Card2(int(faceUpCardind%13), int(faceUpCardind//13))
                newCards = list(currHand)
                newCards.append(faceUpCard)
                for meld in GinRummyUtil.cardsToAllMelds(newCards):
                    if faceUpCard in meld and 3 in state['legal_actions']:
                        print('pick up face up')
                        # pick up card
                        return 3 
                if 2 in state['legal_actions']:
                    # print('draw from deck')
                    return 2
            # 2. Discard highest card in hand that is not apart of any melds
            # 3. Knock if deadwood is less than or equal to 10
            elif len(currHand) == 11:
                minDeadwood = float('inf')
                candidateCards = []
                for card in currHand:
                    remainingCards = list(currHand)
                    remainingCards.remove(card)
                    bestMeldSets = GinRummyUtil.cardsToBestMeldSets(remainingCards)
                    deadwood = GinRummyUtil.getDeadwoodPoints3(remainingCards) if len(bestMeldSets) == 0 \
                        else GinRummyUtil.getDeadwoodPoints1(bestMeldSets[0], remainingCards)
                    if deadwood <= minDeadwood:
                        if deadwood < minDeadwood:
                            minDeadwood = deadwood
                            candidateCards.clear()
                        candidateCards.append(card)
                discard = candidateCards[randint(0, len(candidateCards)-1)]
                # print(minDeadwood)
                if minDeadwood <= 10 and (discard.getId() + 58) in state['legal_actions']:
                    # Knock
                    # print('Knock', discard.getId())
                    # print('Knock', discard)
                    return discard.getId() + 58
                else:
                    # Discard
                    # print('Discard', discard.getId())
                    # print('Discard', discard)
                    return discard.getId() + 6
        # state is 4, declare dead hand (or other, decide randomly)
        return np.random.choice(state['legal_actions'])

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the Simple Gin Rummy agent is not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        probs = [0 for _ in range(self.action_num)]
        for i in state['legal_actions']:
            probs[i] = 1/len(state['legal_actions'])
        return self.step(state), probs
