#include <bits/stdc++.h>
using namespace std;

inline int read()
{
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

inline long long readll() {
	long long x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

const int N=100002;
int x,y,a[N],n,i,ans;
vector<int>e[N];
unordered_map<int,int>mp[N];
void dfs(int u,int fa){
	int mx=1;
	a[u]^=a[fa];
	for (auto v:e[u])
		if (v!=fa){
			dfs(v,u);
			if (mp[v].size()>mp[u].size()) swap(mp[u],mp[v]);
			for (auto [x,y]:mp[v]) mx=max(mx,mp[u][x]+=y);
		}
	if (u!=1 && e[u].size()==1) mp[u][a[u]]=1,ans++;
	else if (mx>1){
		ans-=mx-1;
		for (auto it=mp[u].begin();it!=mp[u].end();)
			if (it->second!=mx) it=mp[u].erase(it);
			else it->second=1,it++;
	}
}
int main(){
	ios::sync_with_stdio(false),cin.tie(0);
	n = read();
	for (i=1;i<=n;i++) a[i] = read();
	for (i=1;i<n;i++){
	    x = read(), y = read();
		e[x].push_back(y),e[y].push_back(x);
	}
	dfs(1,0);
	cout<<ans-mp[1].count(0)<<endl;
}