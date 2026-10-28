# Data-Science-Project-Profolio

#include <iostream>
#include <list>
#include <string>
#include <algorithm>
#include <random>

using namespace std;

// Card structure
struct Card
{
    string name;
    int value;
};

// Calculate hand total modulo 10
int handTotal(const list<Card>& hand)
{
    int total = 0;

    for (Card card : hand)
    {
        total += card.value;
    }

    return total % 10;
}

// Format and return game result
string gameResult(const list<Card>& player, const list<Card>& banker)
{
    string playerCards = "";
    string bankerCards = "";

    bool first = true;

    for (Card card : player)
    {
        if (!first)
            playerCards += ",";

        playerCards += card.name;
        first = false;
    }

    first = true;

    for (Card card : banker)
    {
        if (!first)
            bankerCards += ",";

        bankerCards += card.name;
        first = false;
    }

    int pTotal = handTotal(player);
    int bTotal = handTotal(banker);

    string outcome;

    if (pTotal > bTotal)
    {
        outcome = "PLAYER";
    }
    else if (bTotal > pTotal)
    {
        outcome = "BANKER";
    }
    else
    {
        outcome = "TIE";
    }

    return "Phand-" + playerCards +
           "- " + outcome +
           " - Bhand-" + bankerCards;
}

// Create 8-deck shoe
list<Card> createDeck()
{
    list<Card> deck;

    for (int shoe = 0; shoe < 8; shoe++)
    {
        for (int suit = 0; suit < 4; suit++)
        {
            // Cards 1-9
            for (int card = 1; card <= 9; card++)
            {
                deck.push_back({to_string(card), card});
            }

            // 10, J, Q, K count as 0
            deck.push_back({"10", 0});
            deck.push_back({"J", 0});
            deck.push_back({"Q", 0});
            deck.push_back({"K", 0});
        }
    }

    return deck;
}

// Prepare shoe
list<Card> prepareShoe()
{
    list<Card> deck = createDeck();

    // Convert list to vector for shuffling
    vector<Card> temp(deck.begin(), deck.end());

    random_device rd;
    mt19937 g(rd());

    shuffle(temp.begin(), temp.end(), g);

    deck.assign(temp.begin(), temp.end());

    // Burn card
    Card cutCard = deck.front();
    deck.pop_front();

    int burn;

    if (cutCard.value == 0)
    {
        burn = 10;
    }
    else
    {
        burn = cutCard.value;
    }

    for (int i = 0; i < burn; i++)
    {
        deck.pop_front();
    }

    return deck;
}

// Main game function
string playGame()
{
    list<Card> deck = prepareShoe();

    list<Card> player;
    list<Card> banker;

    // Initial deal
    player.push_back(deck.front());
    deck.pop_front();

    player.push_back(deck.front());
    deck.pop_front();

    banker.push_back(deck.front());
    deck.pop_front();

    banker.push_back(deck.front());
    deck.pop_front();

    int pTotal = handTotal(player);
    int bTotal = handTotal(banker);

    // Natural 8 or 9
    if (pTotal == 8 || pTotal == 9 ||
        bTotal == 8 || bTotal == 9)
    {
        return gameResult(player, banker);
    }

    // Player stands on 6 or 7
    if (pTotal == 6 || pTotal == 7)
    {
        if (bTotal <= 5)
        {
            banker.push_back(deck.front());
            deck.pop_front();
        }

        return gameResult(player, banker);
    }

    // Player draws third card
    Card playerThird = deck.front();
    deck.pop_front();

    player.push_back(playerThird);

    int thirdValue = playerThird.value;

    // Banker drawing rules
    if (bTotal <= 2)
    {
        banker.push_back(deck.front());
        deck.pop_front();
    }
    else if (bTotal == 3)
    {
        if (thirdValue != 8)
        {
            banker.push_back(deck.front());
            deck.pop_front();
        }
    }
    else if (bTotal == 4)
    {
        if (thirdValue >= 2 && thirdValue <= 7)
        {
            banker.push_back(deck.front());
            deck.pop_front();
        }
    }
    else if (bTotal == 5)
    {
        if (thirdValue == 4 ||
            thirdValue == 5 ||
            thirdValue == 6 ||
            thirdValue == 7)
        {
            banker.push_back(deck.front());
            deck.pop_front();
        }
    }
    else if (bTotal == 6)
    {
        if (thirdValue == 6 ||
            thirdValue == 7)
        {
            banker.push_back(deck.front());
            deck.pop_front();
        }
    }

    return gameResult(player, banker);
}

int main()
{
    cout << playGame() << endl;

    return 0;
}
