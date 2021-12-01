#include "../libraries/aoc.h"

map<pair<deque<ll>,deque<ll>>, Player> memo;

ll score(Game g) {
    ll ans = 0;
    ll x = g.winner.deck.cards.size();
    while (x>0) {
        ans += (g.winner.deck.cards.front()*(x--));
        g.winner.deck.cards.pop_front();
    }
    return ans;
}

Player playGame(Game g, map<pair<deque<ll>,deque<ll>>,int> visitedDeck) {
    Player p1 = g.players[0];
    Player p2 = g.players[1];
    ll card1, card2;
    Player winner;
    pair<deque<ll>,deque<ll>> p;
    while (!p1.deck.cards.empty() && !p2.deck.cards.empty()) {
        p.first=p1.deck.cards;
        p.second=p2.deck.cards;
        if (visitedDeck[p]>0) {
            return p1;
        }
        visitedDeck[p]=1;
        card1 = p1.deck.cards.front();
        card2 = p2.deck.cards.front();
        p1.deck.cards.pop_front();
        p2.deck.cards.pop_front();
        if (p1.deck.cards.size()>=card1 && p2.deck.cards.size()>=card2) {
            Game g2;
            Player p3, p4;
            for (int i = 0 ; i < card1; i++) {
                p3.deck.cards.push_back(p1.deck.cards[i]);
            }
            for (int j = 0 ; j < card2; j++) {
                p4.deck.cards.push_back(p2.deck.cards[j]);
            }
            p3.id=p1.id;
            p4.id=p2.id;
            g2.players.push_back(p3);
            g2.players.push_back(p4);
            winner = playGame(g2,{});
        } else {
            if (card1>card2) {
                winner= p1;
            } else {
                winner= p2;
            }
        }
        if (winner.id==p1.id) {
            p1.deck.cards.push_back(card1);  
            p1.deck.cards.push_back(card2);
        } else {
            p2.deck.cards.push_back(card2);
            p2.deck.cards.push_back(card1);
        }
    }
    if (p1.deck.cards.empty()) {
        winner = p2;
    } else {
        winner = p1;
    }
    return winner;
}

int main() {
    auto start = high_resolution_clock::now();
    freopen("inputs/big.txt","r",stdin);
    string line;
    deque<ll> cards;
    int i = 0, player;
    Game g;
    Player p;
    Deck d;
    while (getline(cin,line)) {
        if (line.empty()) {
            p.id=player;
            d.cards=cards;
            p.deck=d;
            g.players.push_back(p);
            cards.clear();
            i=0;
            continue;
        }
        if (i==0) {
            player=stoi(line.substr(7,line.find(':')-7));
        } else {
            ll value = stoi(line);
            cards.push_back(value);
        }
        i++;
    }
    p.id=player;
    d.cards=cards;
    p.deck=d;
    g.players.push_back(p);

    // part 2
    g.winner = playGame(g,{});
    cout<<score(g)<<endl;
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop-start);
    cout<<duration.count()<<endl;
    return 0;
}