# -------------------------------------------------------------------------------
#  RandGinRummyPlayer
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
import random

# Import MLP Models
# from SupervisedLearning.models import *

Card = TypeVar('Card')

class RandGinRummyPlayer(GinRummyPlayer):

    # Inform player of 0-based player number (0/1), starting player number (0/1), and dealt cards
    def startGame(self, playerNum: int, startingPlayerNum: int, cards: List[Card]) -> None:
        self.playerNum = playerNum
        self.startingPlayerNum = startingPlayerNum
        self.cards = list(cards)
        self.opponentKnocked = False
        self.drawDiscardBitstrings = [] # long[], or List[int]
        self.faceUpCard = None
        self.drawnCard = None
        self.state = None

    def willDrawFaceUpCard(self, card: Card) -> bool:
        # Return random choice
        self.faceUpCard = card
        newCards = list(self.cards)
        newCards.append(card)
        choice = random.randint(0, 1)
        if choice == 0:
            return True
        return False


    # Report that the given player has drawn a given card and, if known, what the card is.
    # If the card is unknown because it is drawn from the face-down draw pile, the drawnCard is null.
    # Note that a player that returns false for willDrawFaceUpCard will learn of their face-down draw from this method.
    def reportDraw(self, playerNum: int, drawnCard: Card) -> None:
        # Ignore other player draws.  Add to cards if playerNum is this player.
        if playerNum == self.playerNum:
            self.cards.append(drawnCard)
            self.drawnCard = drawnCard

    # Get the player's discarded card.  If you took the top card from the discard pile,
    # you must discard a different card.
    # If this is not a card in the player's possession, the player forfeits the game.
    # @return the player's chosen card for discarding
    def getDiscard(self) -> Card:

        choice = random.randint(0, len(self.cards)-1)
        discCard = self.cards[choice]
        while discCard == self.faceUpCard:
            choice = random.randint(0, len(self.cards)-1)
            discCard = self.cards[choice]
        return discCard


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