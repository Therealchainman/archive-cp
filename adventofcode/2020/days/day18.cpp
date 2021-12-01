#include "../libraries/aoc.h"

ll computeRow(string expr, bool prec) {
    stack<char> operatorStack;
    stack<ll> operandStack;
    ll a, b;
    for (char oper : expr) {
        if (isdigit(oper)) {
            operandStack.push(oper-'0');
        } else if (oper=='(') {
            operatorStack.push('(');
        } else if (oper==')') {
            while (!operatorStack.empty() && operatorStack.top()!='(') {
                a = operandStack.top();
                operandStack.pop();
                b = operandStack.top();
                operandStack.pop();
                operandStack.push(evaluate(a,b,operatorStack.top()));
                operatorStack.pop();
            }
            operatorStack.pop();
        } else if (oper == '+' || oper=='*') {
            while (!operatorStack.empty() && operatorStack.top() != '(' && (!prec || prec && oper=='*' && operatorStack.top()=='+')) {
                a = operandStack.top();
                operandStack.pop();
                b = operandStack.top();
                operandStack.pop();
                operandStack.push(evaluate(a,b,operatorStack.top()));
                operatorStack.pop();
            }
            operatorStack.push(oper);
        }
    }
    return operandStack.top();
}


ll solve(number n,bool prec) {
    ll ans = 0;
    for (string line : n.lines) {
        ans+=computeRow(line,prec);
    }
    return ans;
}

int main() {
    freopen("inputs/big.txt","r",stdin);
    number n;
    string input,tmp,operands;
    while (getline(cin,input)) {
        istringstream s(input);
        operands = "(";
        while (getline(s,tmp,' ')) {
            operands+=tmp;
        }
        operands+=")";
        n.lines.push_back(operands);
    }
    int start = 0;
    // part 1
    cout<<solve(n,false)<<endl;
    // part 2
    cout<<solve(n,true)<<endl;
    return 0;
}