#include "../libraries/aoc.h"

int main() {
    freopen("inputs/big.txt","r",stdin);
    map<string, set<string>> allergensToIngredients;
    multiset<string> allIngredients;
    string curLine;
    while (getline(cin,curLine)) {
        stringstream lineStream(curLine);
        set<string> curIngredients;
        set<string> curAllergens;
        string curToken;
        while (lineStream >> curToken) {
            if (curToken[0] == '(') {
                break;
            } else {
                curIngredients.insert(curToken);
            }
        }
        allIngredients.insert(curIngredients.begin(),curIngredients.end());
        while (lineStream >> curToken) {
            const auto curAllergen = curToken.substr(0,curToken.size()-1);
            curAllergens.insert(curAllergen);
            if (allergensToIngredients.count(curAllergen)) {
                auto& otherIngredients = allergensToIngredients.at(curAllergen);
                set<string> intersection;
                set_intersection(otherIngredients.cbegin(),otherIngredients.cend(), curIngredients.cbegin(), curIngredients.cend(), inserter(intersection, intersection.begin()));
                otherIngredients = move(intersection);
            } else {
                allergensToIngredients.insert({curAllergen, curIngredients});
            }
        }
    }
    while (any_of(allergensToIngredients.cbegin(),allergensToIngredients.cend(), [] (auto& it) {
        return it.second.size() > 1;})
    )  {
        for (const auto& allergenToIngredientA : allergensToIngredients) {
            if (allergenToIngredientA.second.size()==1) {
                const auto& toRemove = *allergenToIngredientA.second.begin();
                for (auto& allergenToIngredientB : allergensToIngredients) {
                    if (allergenToIngredientA.first == allergenToIngredientB.first) {
                        continue;
                    }
                    allergenToIngredientB.second.erase(toRemove);
                }
            }
        }
    }
    uintmax_t Part1{};
    {
        multiset<string> safeIngredients = allIngredients;
        for (const auto& allergicIngredient : allergensToIngredients) {
            safeIngredients.erase(*allergicIngredient.second.begin());
        } 
        Part1 = safeIngredients.size();
    }
    cout<< Part1<<endl;
    string Part2;
    {
        for (const auto& allergenToIngredient : allergensToIngredients) {
            Part2 += *allergenToIngredient.second.begin() + ',';
        }
        Part2.pop_back();
    }
    cout<<Part2<<endl;
}