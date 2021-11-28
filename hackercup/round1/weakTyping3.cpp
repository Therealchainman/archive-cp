#include <bits/stdc++.h>
using namespace std;
const int MOD = 1e9+7;
/*
inline is useful for functions that are small, but the function call is overhead you don't want, 
that is the function call is greater than the actual execution of function code. 
I actually removed my inline code so this looks like a dumb statement but it was originally for a min 
function. 

So only O and X actually matter, anytime you see one of these characters and they are different than the previous
that means you have to update the delta which is the amount of change for the sum at each index going forward. 
So it is storing the amount of swaps requires for each substring prior. 

Really I saw the solution to this by writing out an example and then I saw how it increased a factor that was like a delta or change
that was then used for each string considered prior
That is you consider all these strings S[0:i], and like there are n substrings in here, but I already know the number of swaps required
by 
*/
/*
On the matrix exponentiation solution to this problem.  This was a good problem to practice
using matrix exponentiation. 

The first step is to derive the transition matrices.  I think for this one
you will have 3 transition matricies for F, X, O
so like T_F, T_X, T_O
So as you iterate through you build up like this 
take the string XFO..
so you get (T_F*T_X*T_O)^3B = A
what about XFO..FO.
((T_F*T_X*T_O)^3*T_F*T_O)^2B = A

This is the idea so really just need to find these transition matrices

Matrix to a power can be solved in logarithmic.  T^n can be sovled in Olog(n)
*/
struct Matrix {
    int numRows, numCols;
    vector<vector<int>> M;
    // initialize the 2-dimensional array representation for the matrix with 
    // a given value. 
    void init(int r, int c, int val) {
        numRows = r, numCols = c;
        M.resize(r);
        for (int i = 0;i<r;i++) {
            M[i].assign(c, val);
        }
    }
    // neutral matrix is just one's along the main diagonal (identity matrix)
    void neutral(int r, int c) {
        numRows = r, numCols = c;
        M.resize(r);
        for (int i = 0;i<r;i++) {
            M[i].assign(c, 0);
        }
        for (int i = 0;i<r;i++) {
            for (int j = 0; j < c;j++) {
                if (i==j) {
                    M[i][j]=1;
                }
            }
        }

    }
    // Set's a pair of coordinates on the matrix with the specified value, works for a transition matrix
    // where you need ones in places. 
    void set(vector<pair<int,int>> locs, int val) {
        int r, c;
        for (auto loc : locs) {
            tie(r, c) = loc;
            M[r][c] = val;
        }
    }
    // this matrix times another matrix. 
    void operator*=(const Matrix& B) {
        int RB = B.M.size(), CB = B.M[0].size();
        vector<vector<int>> result(numRows, vector<int>(CB, 0));
        for (int i = 0;i < numRows;i++) {
            for (int j = 0;j < CB;j++) {
                int sum = 0;
                for (int k = 0;k < RB;k++) {
                    sum = (sum + ((long long)M[i][k]*B.M[k][j])%MOD)%MOD;
                }
                result[i][j] = sum;
            }
        }
        numRows = numCols, numCols = RB;
        swap(M, result);
    }

    void transpose() {
        int R = numCols, C = numRows;
        vector<vector<int>> matrix(R, vector<int>(C,0));
        for (int i = 0;i < numRows;i++) {
            for (int j = 0;j < numCols;j++) {
                matrix[j][i]=M[i][j];
            }
        }
        swap(numRows,numCols); // transpose swaps the rows and columns
        swap(M,matrix); // swap these two
    }

    // Method to convert a row and column to a unique integer that identifies a row, column combination
    // that can be used in hashing
    int getId(int row, int col) {
        return numRows*row+col;
    }
};

Matrix operator*(const Matrix& A, const Matrix& B) {
    int RA = A.M.size(), CA = A.M[0].size(), RB = B.M.size(), CB = B.M[0].size();
    if (CA!=RB) {
        printf("CA and RB are not equal\n");
        return A;
    }
    Matrix result;
    result.init(RA,CB,0);
    for (int i = 0;i < RA;i++) {
        for (int j = 0; j < CB; j++) {
            int sum = 0;
            for (int k = 0;k < RB;k++) {
                sum = (sum+((long long)A.M[i][k]*B.M[k][j])%MOD)%MOD;
            }
            result.M[i][j]=sum;
        }
    }
    return result;
}

// matrix exponentiation
Matrix matrix_power(Matrix& A, int b) {
    Matrix result;
    result.neutral(A.numRows, A.numCols);
    while (b > 0) {
        if (b % 2 == 1) {
            result = (result*A);
        }
        A *= A;
        b /= 2;
    }
    return result;
}

// int main() {
//     // example of how to use matrix exponentiation
//     // This will not really work as it doesn't have all the inputs.  
//     int n;
//     Matrix transition, base;
//     transition.init(5, 5, 0);
//     base.init(5, 1, 1);
//     vector<pair<int,int>> ones = {{0,1}, {0, 2}, {0, 4}, {1, 0}, {1, 2}, {2, 1}, {2, 3}, {3, 2}, {4, 2}, {4, 3}};
//     transition.set(ones, 1);
//     Matrix expo = matrix_power(transition, n-1); // exponentiated transition matrix
//     Matrix result = expo*base;
//     result.transpose();
//     int sum = 0;
//     for (int i = 0;i<5;i++) {
//         sum = (sum+result.M[0][i])%MOD;
//     }
//     return sum;
// }

int main() {
    int T;
    // freopen("inputs/weak_typing_chapter_2_validation_input.txt", "r", stdin);
    // freopen("outputs/weak_typing_chapter_2_validation_output.txt", "w", stdout);
    freopen("inputs/weak_typing_chapter_2_input.txt", "r", stdin);
    freopen("outputs/weak_typing_chapter_2_output.txt", "w", stdout);
    cin>>T;
    for (int t = 1;t<=T;t++) {
        int N;
        string S;
        cin>>N>>S;
        int sum = 0, last = -1, delta = 0; 
        char prev = 'F';
        for (int i = 0;i<N;i++) {   
            if (S[i]=='F') {
                newOs = 0*Xs + 1*Os + 0*Fs + 0*sum + 0*delta + 0*addOne;
                newXs = 1*Xs + 0*Os + 0*Fs + 0*sum + 0*delta + 0*addOne;
                newFs = 0*Xs + 0*Os + 1*Fs + 0*sum + 0*delta + 1*addOne;
                newDelta = 0*Xs + 0*Os + 0*Fs + 0*sum + 1*delta + 0*addOne;
                newSum = 0*Xs + 0*Os + 0*Fs + 1*sum + 1*delta + 0*addOne;
                newOne = addOne
            } else if (S[i]=='O') {
                newOs = 1*Xs + 1*Os + 1*Fs + 0*sum + 0*delta + 1*addOne;
                newXs = 0*Xs + 0*Os + 0*Fs + 0*sum + 0*delta + 0*addOne;
                newFs = 0*Xs + 0*Os + 0*Fs + 0*sum + 0*delta + 0*addOne;
                newDelta = 1*Xs + 0*Os + 0*Fs + 0*sum + 1*delta + 0*addOne;
                newSum = 1*Xs + 0*Os + 0*Fs + 1*sum + 1*delta + 0*addOne;
            } else {
                newOs = 0*Xs + 0*Os + 0*Fs + 0*sum + 0*delta + 0*addOne;
                newXs = 1*Xs + 1*Os + 1*Fs + 0*sum + 0*delta + 1*addOne;
                newFs = 0*Xs + 0*Os + 0*Fs + 0*sum + 0*delta + 0*addOne;
                newDelta = 0*Xs + 1*Os + 0*Fs + 0*sum + 1*delta + 0*addOne;
                newSum = 0*Xs + 1*Os + 0*Fs + 1*sum + 1*delta + 0*addOne;            
            }

            if (S[i]=='F' || S[i]==prev) {
                if (S[i]!='F') {
                    last=i;
                }
            } else {
                delta = (delta+last+1)%MOD;
                last = i;
            }
            sum = (sum + delta)%MOD;
            prev = S[i]=='F' ? prev : S[i];
        }
        printf("Case #%d: %d\n", t, sum);
    }
}