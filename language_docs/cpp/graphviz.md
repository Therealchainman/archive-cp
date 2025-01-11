# Graphviz

Using the graphviz library to visualize graphs in C++. 

Some starter code to refresh. 

```cpp
#include <gvc.h>
#include <cstdlib>
GVC_t *gvc = gvContext();
Agraph_t *g = agopen("G", Agdirected, nullptr);
vector<Agnode_t*> nodes;
for (int i = 0; i < R + C; i++) {
    string nodeName = to_string(i);
    Agnode_t *n = agnode(g, const_cast<char*>(nodeName.c_str()), 1);
    nodes.emplace_back(n);
}
for (int r = 0; r < R; r++) {
    for (int c = 0; c < C; c++) {
        if (isSet(B[r][c], i)) {
            agedge(g, nodes[r], nodes[c + R], nullptr, 1);
            adj[r].emplace_back(c + R);
        } else {
            agedge(g, nodes[c + R], nodes[r], nullptr, 1);
            adj[c + R].emplace_back(r);
        }
    }
}
gvLayout(gvc, g, "dot");
string filename = "output/graph" + to_string(i) + ".png";
gvRenderFilename(gvc, g, "png", filename.c_str());
gvFreeLayout(gvc, g);
agclose(g);
gvFreeContext(gvc);
cout << "Graph rendered to " << filename << endl;
```