''' Game-related and Env-related base classes
'''

class Card(object):
    '''
    Card stores the suit and rank of a single card

    Note:
        The suit variable in a standard card game should be one of [S, H, D, C, BJ, RJ] meaning [Spades, Hearts, Diamonds, Clubs, Black Joker, Red Joker]
        Similarly the rank variable should be one of [A, 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K]
    '''

    suit = None
    rank = None
    valid_suit = ['S', 'H', 'D', 'C', 'BJ', 'RJ']
    valid_rank = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']

    def __init__(self, suit, rank):
        ''' Initialize the suit and rank of a card

        Args:
            suit: string, suit of the card, should be one of valid_suit
            rank: string, rank of the card, should be one of valid_rank
        '''
        self.suit = suit
        self.rank = rank

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        else:
            # don't attempt to compare against unrelated types
            return NotImplemented

    def __hash__(self):
        suit_index = Card.valid_suit.index(self.suit)
        rank_index = Card.valid_rank.index(self.rank)
        return rank_index + 100 * suit_index

    def __str__(self):
        ''' Get string representation of a card.

        Returns:
            string: the combination of rank and suit of a card. Eg: AS, 5H, JD, 3C, ...
        '''
        return self.rank + self.suit

    def get_index(self):
        ''' Get index of a card.

        Returns:
            string: the combination of suit and rank of a card. Eg: 1S, 2H, AD, BJ, RJ...
        '''
        return self.suit+self.rank


class Card2:

    # Array of abbreviated card rank names in ascending order of rank.
    rankNames = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]

    # Array of abbreviated card suit names indexed by suit index.
    # Changed the suits to match rlcard
    suitNames = ['S', 'H', 'D', 'C']#["C", "H", "S", "D"]

    # Number of card ranks.
    NUM_RANKS = len(rankNames)

    # Number of card suits.
    NUM_SUITS = len(suitNames)

    # Total number of cards.
    NUM_CARDS = NUM_RANKS * NUM_SUITS

    # Constructor to create a card object with the corresponding zero-based indices to rankNames and suitNames, respectively.
    # AVOID USE IF POSSIBLE.  Use the Card objects already created in allCards, retrieving them via
    # (1) strCardMap using the method get(String),
    # (2) getCard(int), or
    # (3) getCard(int rank, int suit).
    # @param rank rank of card (zero-based index to rankNames)
    # @param suit suit of card (zero-based index to suitNames)
    def __init__(self, rank, suit):
       # rank index (zero-based index to rankNames)
        self.rank = rank

        # suit index (zero-based index to suitNames)
        self.suit = suit

    # Get rank of card (zero-based index to rankNames).
    # @return rank of card (zero-based index to rankNames)
    def getRank(self):
        return self.rank

    # Get suit of card (zero-based index to suitNames).
    # @return suit of card (zero-based index to suitNames)
    def getSuit(self):
        return self.suit

    # Return whether or not the card is Red.
    # @return whether or not the card is Red
    def isRed(self):
        return self.suit % 2 == 1

    # Return the Card id number.
    # @return the Card id number
    def getId(self):
        return self.suit * Card2.NUM_RANKS + self.rank

    # Return card representation as a string.
    def __str__(self):
        return Card2.rankNames[self.rank] + Card2.suitNames[self.suit]

    def __repr__(self):
        return Card2.rankNames[self.rank] + Card2.suitNames[self.suit]



import random
from sys import stderr, exit

class Deck:

    # An array of all unique Card objects.
    allCards = []

    # Array of abbreviated card rank names in ascending order of rank.
    rankNames = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]

    # Array of abbreviated card suit names indexed by suit index.
    # Changed the suits to match rlcard
    suitNames = ['S', 'H', 'D', 'C']#["C", "H", "S", "D"]
    
    # Parallel array to suitNames indicating whether or not the corresponding suit is red.
    isSuitRed = [False, True, True, False]#[False, True, False, True]

    # Number of card ranks.
    NUM_RANKS = len(rankNames)

    # Number of card suits.
    NUM_SUITS = len(suitNames)

    # Total number of cards.
    NUM_CARDS = NUM_RANKS * NUM_SUITS

    # Map from String representations to Card objects.
    strCardMap = {}

    # Map from String representations to Card id numbers.
    strIdMap = {}

    # Map from Card id numbers to String representations.
    idStrMap = {}

    # Create all cards and initialize static maps.
    # for s in suitNames:
        # for r in rankNames:
    for s in range(NUM_SUITS):
        for r in range(NUM_RANKS):
            c = Card2(r, s)
            allCards += [c]
            strCardMap[str(c)] = c
            strIdMap[str(c)] = c.getId()
            idStrMap[c.getId()] = str(c)

    # Get a Deck of cards without a shuffle
    # @return a deck of cards in standard order
    def getDeck():
        return Deck.allCards

    # Get the Card object corresponding to the given Card id number (0 - 51)
    # @param id Card id number (0 - 51)
    # @param rank index
    # @param suit index
    # @return corresponding Card object
    def getCard(id=-1, rank=-1, suit=-1):
        if id >= 0:
            return Deck.allCards[id]
        elif rank >= 0 and suit >= 0:
            return Deck.allCards[suit * Deck.NUM_RANKS + rank]
        else:
            print("ERROR: bad parameters to getCard in Deck class.", file=stderr)
            exit(1)

    # Get the Card id number corresponding to the given rank and suit indices
    # @param rank rank index
    # @param suit suit index
    # @return corresponding Card id number
    def getId(rank, suit):
        return suit * Deck.NUM_RANKS + rank

    # Return a Stack deck of Cards corresponding to the given shuffle seed number
    # @param seed shuffle seed number
    # @return corresponding Stack deck of Cards
    def getShuffle(seed):
        deck = []
        random.seed(seed)
        for i in range(Deck.NUM_CARDS):
            deck += [Deck.allCards[i]]
        random.shuffle(deck)
        return deck


#######################################################################################################################################

class Dealer(object):
    ''' Dealer stores a deck of playing cards, remained cards
    holded by dealer, and can deal cards to players

    Note: deck variable means all the cards in a single game, and should be a list of Card objects.
    '''

    deck = []
    remained_cards = []

    def __init__(self):
        ''' The dealer should have all the cards at the beginning of a game
        '''
        raise NotImplementedError

    def shuffle(self):
        ''' Shuffle the cards holded by dealer(remained_cards)
        '''
        raise NotImplementedError

    def deal_cards(self, **kwargs):
        ''' Deal specific number of cards to a specific player

        Args:
            player_id: the id of the player to be dealt cards
            num: number of cards to be dealt
        '''
        raise NotImplementedError

class Player(object):
    ''' Player stores cards in the player's hand, and can determine the actions can be made according to the rules
    '''

    player_id = None
    hand = []

    def __init__(self, player_id):
        ''' Every player should have a unique player id
        '''
        self.player_id = player_id

    def available_order(self):
        ''' Get the actions can be made based on the rules

        Returns:
            list: a list of available orders
        '''
        raise NotImplementedError

    def play(self):
        ''' Player's actual action in the round
        '''
        raise NotImplementedError

class Judger(object):
    ''' Judger decides whether the round/game ends and return the winner of the round/game
    '''

    def judge_round(self, **kwargs):
        ''' Decide whether the round ends, and return the winner of the round

        Returns:
            int: return the player's id who wins the round or -1 meaning the round has not ended
        '''
        raise NotImplementedError

    def judge_game(self, **kwargs):
        ''' Decide whether the game ends, and return the winner of the game

        Returns:
            int: return the player's id who wins the game or -1 meaning the game has not ended
        '''
        raise NotImplementedError


class Round(object):
    ''' Round stores the id the ongoing round and can call other Classes' functions to keep the game running
    '''

    def __init__(self):
        ''' When the game starts, round id should be 1
        '''

        raise NotImplementedError

    def proceed_round(self, **kwargs):
        ''' Call other Classes's functions to keep the game running
        '''
        raise NotImplementedError


class Game(object):
    ''' Game class. This class will interact with outer environment.
    '''

    def init_game(self):
        ''' Initialize all characters in the game and start round 1
        '''
        raise NotImplementedError

    def step(self, action):
        ''' Perform one draw of the game and return next player number, and the state for next player
        '''
        raise NotImplementedError

    def step_back(self):
        ''' Takes one step backward and restore to the last state
        '''
        raise NotImplementedError

    def get_player_num(self):
        ''' Retrun the number of players in the game
        '''
        raise NotImplementedError

    def get_action_num(self):
        ''' Return the number of possible actions in the game
        '''
        raise NotImplementedError

    def get_player_id(self):
        ''' Return the current player that will take actions soon
        '''
        raise NotImplementedError

    def is_over(self):
        ''' Return whether the current game is over
        '''
        raise NotImplementedError

