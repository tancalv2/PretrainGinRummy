# -------------------------------------------------------------------------------
#  MLPGinRummyPlayer
#
#  This estimation will be calculated using a Multilayer Percepton trained on the
#  SimpleGinRummyPlayer written
#  by Calvin Tan.
#
#  @author Calvin Tan
#  @version 1.0
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# The following code was originally written by Todd Neller in Java.
# It was translated into Python by May Jiang.
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# Copyright (C) 2020 Todd Neller
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# Information about the GNU General Public License is available online at:
#   http://www.gnu.org/licenses/
# To receive a copy of the GNU General Public License, write to the Free
# Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
# 02111-1307, USA.
# -------------------------------------------------------------------------------

from typing import List, TypeVar
from random import randint
from GinRummyUtil import GinRummyUtil
from GinRummyPlayer import GinRummyPlayer

# Import MLP Models
# from SupervisedLearning.models import *

Card = TypeVar('Card')

class MLPGinRummyPlayer(GinRummyPlayer):

    def loadModel(self, model_pt):
        print('Load Model')
        self.model = model_pt

    def setVerbose(self, verbose):
        self.playVerbose = verbose

    def updateStates(self, states):
        if self.playVerbose:
            print('Update States')
        self.state = states

    def knockAction(self) -> bool:
        return self.knock

    # Inform player of 0-based player number (0/1), starting player number (0/1), and dealt cards
    def startGame(self, playerNum: int, startingPlayerNum: int, cards: List[Card]) -> None:
        self.playerNum = playerNum
        self.startingPlayerNum = startingPlayerNum
        self.cards = list(cards)
        self.opponentKnocked = False
        self.drawDiscardBitstrings = [] # long[], or List[int]
        self.faceUpCard = None
        self.faceUpCardBool = False
        self.drawnCard = None
        self.state = None
        self.knock = False
        self.playVerbose = False

    # Return whether or not player will draw the given face-up card on the draw pile.
    def willDrawFaceUpCard(self, card: Card) -> bool:
        self.faceUpCard = card
        # BPBD, either draw(2)->False or pickup(3)->True
        state = np.expand_dims(self.state, axis=0)
        state = torch.from_numpy(state).type(torch.FloatTensor).to(device)
        action = self.model(state)
        action = action.detach().numpy().reshape(-1)
        if self.playVerbose:
            print('Draw new card:', action[2])
            print('Pickup from discard:', action[3])
        if action[3] > action[2]:
            # print('Pickup Discard Action')
            self.faceUpCardBool = True
            return True
        # print('Draw from Deck Action')
        self.faceUpCardBool = False
        return False

    # Report that the given player has drawn a given card and, if known, what the card is.
    # If the card is unknown because it is drawn from the face-down draw pile, the drawnCard is null.
    # Note that a player that returns false for willDrawFaceUpCard will learn of their face-down draw from this method.
    def reportDraw(self, playerNum: int, drawnCard: Card) -> None:
        # Ignore other player draws.  Add to cards if playerNum is this player.
        if playerNum == self.playerNum:
            self.cards.append(drawnCard)
            self.drawnCard = drawnCard






    # def getDiscard(self) -> Card:
    #     # Discard a random card (not just drawn face up) leaving minimal deadwood points.
    #     minDeadwood = float('inf')
    #     candidateCards = []
    #     for card in self.cards:
    #         # Cannot draw and discard face up card.
    #         if card == self.drawnCard and self.drawnCard == self.faceUpCard:
    #         # if card == self.drawnCard and self.faceUpCard:
    #             continue
    #         # Disallow repeat of draw and discard.
    #         drawDiscard = [self.drawnCard, card]
    #         if GinRummyUtil.cardsToBitstring(drawDiscard) in self.drawDiscardBitstrings:
    #             continue

    #         remainingCards = list(self.cards)
    #         remainingCards.remove(card)
    #         bestMeldSets = GinRummyUtil.cardsToBestMeldSets(remainingCards)
    #         deadwood = GinRummyUtil.getDeadwoodPoints3(remainingCards) if len(bestMeldSets) == 0 \
    #             else GinRummyUtil.getDeadwoodPoints1(bestMeldSets[0], remainingCards)
    #         if deadwood <= minDeadwood:
    #             if deadwood < minDeadwood:
    #                 minDeadwood = deadwood
    #                 candidateCards.clear()
    #             candidateCards.append(card)
    #     # Prevent future repeat of draw, discard pair.
    #     discard = candidateCards[randint(0, len(candidateCards)-1)]
    #     drawDiscard = [self.drawnCard, discard]
    #     self.drawDiscardBitstrings.append(GinRummyUtil.cardsToBitstring(drawDiscard))
    #     return discard

    # Get the player's discarded card.  If you took the top card from the discard pile,
    # you must discard a different card.
    # If this is not a card in the player's possession, the player forfeits the game.
    # @return the player's chosen card for discarding
    def getDiscard(self) -> Card:
        # APBD, either either discard or knock...
        # determine the allowable actions (which cards can be discarded/knocked on)
        currHand = np.array(self.state[0:52])
        knockCards = np.array(self.state[0:52])
        # if self.playVerbose:
        #     print('Current Hand:', un_one_hot(currHand))
        # disallow discarding PickUp FaceUp/Discarded Card
        if self.faceUpCardBool:
        # if self.drawnCard == self.faceUpCard:
            currHand[self.drawnCard.getId()] = 0
            knockCards[self.drawnCard.getId()] = 0
        
        # prune illegal knock actions
        cardIndex = np.where(knockCards == 1)[0]
        for c in cardIndex:
            remainingCards = list(self.cards)
            remainingCards.remove(Deck.getCard(c))
            bestMeldSets = GinRummyUtil.cardsToBestMeldSets(remainingCards)
            deadwood = GinRummyUtil.getDeadwoodPoints3(remainingCards) if len(bestMeldSets) == 0 \
                else GinRummyUtil.getDeadwoodPoints1(bestMeldSets[0], remainingCards)
            if deadwood > 10:
                knockCards[c] = 0

        state = np.expand_dims(self.state, axis=0)
        state = torch.from_numpy(state).type(torch.FloatTensor).to(device)
        action = self.model(state)
        action = action.detach().numpy().reshape(-1)

        discardMax = max(currHand * action[6:58])
        # knockMax = max(currHand * action[58:110])
        knockMax = max(knockCards * action[58:110])

        if self.playVerbose:
            unmeldedCards = self.cards.copy()
            bestMelds = GinRummyUtil.cardsToBestMeldSets(unmeldedCards)
            if len(bestMelds) > 0:
                melds = bestMelds[0]
                for meld in melds:
                    for card in meld:
                        unmeldedCards.remove(card)
                melds.extend(unmeldedCards)
            else:
                melds = unmeldedCards
            print('Current Hand:', melds)
            if np.argmax(action) > 58:
                # print('Knock', all_classes[np.argmax(action)], '| D:', Deck.getCard(np.argmax(currHand * action[6:58])), '| K:', Deck.getCard(np.argmax(currHand * action[58:])), '|', np.argmax(action))
                print('Knock', all_classes[np.argmax(action)], '| D:', Deck.getCard(np.argmax(currHand * action[6:58])), '| K:', Deck.getCard(np.argmax(knockCards * action[58:])), '|', np.argmax(action))
            else:
                # print('Discard', all_classes[np.argmax(action)], '| D:', Deck.getCard(np.argmax(currHand * action[6:58])), '| K:', Deck.getCard(np.argmax(currHand * action[58:])), '|', np.argmax(action))
                print('Discard', all_classes[np.argmax(action)], '| D:', Deck.getCard(np.argmax(currHand * action[6:58])), '| K:', Deck.getCard(np.argmax(knockCards * action[58:])), '|', np.argmax(action))
            print('MAX:{:.4f}, {:.4f}'.format(discardMax, knockMax))

        if discardMax > knockMax or int(sum(knockCards) == 0):
            if self.playVerbose:
                print('Discard Action')
            self.knock = False
            return Deck.getCard(np.argmax(currHand * action[6:58]))
        else:
            if self.playVerbose:
                print('Knock Action')
            self.knock = True
            # return Deck.getCard(np.argmax(currHand * action[58:]))
            return Deck.getCard(np.argmax(knockCards * action[58:]))




















    # Report that the given player has discarded a given card.
    def reportDiscard(self, playerNum: int, discardedCard: Card) -> None:
        # Ignore other player discards.  Remove from cards if playerNum is this player.
        if playerNum == self.playerNum:
            self.cards.remove(discardedCard)

    # At the end of each turn, this method is called and the player that cannot (or will not) end the round will return a null value.
    # However, the first player to "knock" (that is, end the round), and then their opponent, will return an ArrayList of ArrayLists of melded cards.
    # All other cards are counted as "deadwood", unless they can be laid off (added to) the knocking player's melds.
    # When final melds have been reported for the other player, a player should return their final melds for the round.
    # @return null if continuing play and opponent hasn't melded, or an ArrayList of ArrayLists of melded cards.
    def getFinalMelds(self) -> List[List[Card]]:
        # Check if deadwood of maximal meld is low enough to go out.
        bestMeldSets = GinRummyUtil.cardsToBestMeldSets(self.cards) # List[List[List[Card]]]
        if not self.opponentKnocked and (len(bestMeldSets) == 0 or \
            GinRummyUtil.getDeadwoodPoints1(bestMeldSets[0], self.cards) > \
            GinRummyUtil.MAX_DEADWOOD):
            return None
        if len(bestMeldSets) == 0:
            return []
        return bestMeldSets[randint(0, len(bestMeldSets)-1)]

    # When an player has ended play and formed melds, the melds (and deadwood) are reported to both players.
    def reportFinalMelds(self, playerNum: int, melds: List[List[Card]]) -> None:
        # Melds ignored by simple player, but could affect which melds to make for complex player.
        if playerNum != self.playerNum:
            self.opponentKnocked = True

    # Report current player scores, indexed by 0-based player number.
    def reportScores(self, scores: List[int]) -> None:
        # Ignored by simple player, but could affect strategy of more complex player.
        return

    # Report layoff actions.
    def reportLayoff(self, playerNum: int, layoffCard: Card, opponentMeld: List[Card]) -> None:
        # Ignored by simple player, but could affect strategy of more complex player.
        return

    # Report the final hands of players.
    def reportFinalHand(self, playerNum: int, hand: List[Card]) -> None:
        # Ignored by simple player, but could affect strategy of more complex player.
        return