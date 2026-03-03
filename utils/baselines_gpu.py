import cudf
import cugraph
import torch
import dgl
from torch_geometric.utils import k_hop_subgraph
from typing import List, Tuple, Dict, Optional


class KnowledgeGraphRetrievalBaselinesGPU:

    def __init__(self, triplets: List[Tuple[str, str, str]], backend: Optional[str] = None):
        """
        Initialize GPU baseline with specified backend.

        Args:
            triplets: List of (subject, relation, object) tuples
            backend: One of 'cugraph', 'dgl', 'pyg', or None (lazy loading)
        """
        self.triplets = triplets
        self.backend = backend
        self._build_mappings()

        if backend == 'cugraph':
            self._build_cugraph()
        elif backend == 'dgl':
            self._build_dgl_graph()
        elif backend == 'pyg':
            self._build_pyg_graph()

    def _build_mappings(self):
        entities, relations = set(), set()
        for s, r, o in self.triplets:
            entities.add(s)
            entities.add(o)
            relations.add(r)
        self.entity_to_id = {e: i for i, e in enumerate(sorted(entities))}
        self.id_to_entity = {i: e for e, i in self.entity_to_id.items()}
        self.num_entities = len(entities)
        self.num_relations = len(relations)

    # ---------- cuGraph ----------
    def _build_cugraph(self):
        if not hasattr(self, 'cugraph_G'):
            df = cudf.DataFrame({
                "src": [self.entity_to_id[s] for s, _, o in self.triplets],
                "dst": [self.entity_to_id[o] for _, _, o in self.triplets],
            })
            self.cugraph_G = cugraph.Graph(directed=True)
            self.cugraph_G.from_cudf_edgelist(df, source="src", destination="dst", renumber=False)

    def cugraph_khop(self, start_cuis, hops):
        if hops <= 0 or not start_cuis:
            return []

        if not hasattr(self, 'cugraph_G'):
            self._build_cugraph()

        found, start_set = set(), set(start_cuis)
        for s in start_cuis:
            sid = self.entity_to_id.get(s)
            if sid is None:
                continue
            bfs = cugraph.bfs(self.cugraph_G, start=sid, depth_limit=hops)
            layer = bfs.loc[bfs["distance"] == hops, "vertex"].to_pandas().tolist()
            found.update(self.id_to_entity[i] for i in layer)
        return list(found - start_set)

    # ---------- DGL ----------
    def _build_dgl_graph(self):
        if not hasattr(self, 'dgl_g'):
            src = [self.entity_to_id[s] for s, _, o in self.triplets]
            dst = [self.entity_to_id[o] for _, _, o in self.triplets]

            self.dgl_g = dgl.graph((torch.tensor(src, dtype=torch.int64),
                                    torch.tensor(dst, dtype=torch.int64)),
                                   num_nodes=self.num_entities)
            self.dgl_g = self.dgl_g.to('cuda')

    def dgl_khop(self, start_cuis, hops):
        if hops <= 0 or not start_cuis:
            return []

        if not hasattr(self, 'dgl_g'):
            self._build_dgl_graph()

        start_set = set(start_cuis)
        all_exact_k = set()

        for cui in start_cuis:
            if cui not in self.entity_to_id:
                continue
            seed_id = self.entity_to_id[cui]

            dist_tensor = dgl.shortest_dist(self.dgl_g, root=seed_id, return_paths=False)
            exact_k_mask = (dist_tensor == hops)
            exact_k_nodes = torch.where(exact_k_mask)[0]

            for node_id in exact_k_nodes.cpu().tolist():
                entity_name = self.id_to_entity[node_id]
                if entity_name not in start_set:
                    all_exact_k.add(entity_name)

        return list(all_exact_k)

    # ---------- PyG ----------
    def _build_pyg_graph(self):
        if not hasattr(self, 'pyg_edge_index'):
            src = torch.tensor([self.entity_to_id[s] for s, _, o in self.triplets], dtype=torch.long)
            dst = torch.tensor([self.entity_to_id[o] for _, _, o in self.triplets], dtype=torch.long)
            self.pyg_edge_index = torch.stack([src, dst], dim=0)
            if torch.cuda.is_available():
                self.pyg_edge_index = self.pyg_edge_index.to('cuda')

    def pyg_khop(self, start_cuis, hops):
        if hops <= 0 or not start_cuis:
            return []

        if not hasattr(self, 'pyg_edge_index'):
            self._build_pyg_graph()

        out, start_set = set(), set(start_cuis)
        for s in start_cuis:
            sid = self.entity_to_id.get(s)
            if sid is None:
                continue
            S_k, *_ = k_hop_subgraph(
                sid, hops, self.pyg_edge_index,
                relabel_nodes=False,
                num_nodes=self.num_entities,
                flow='target_to_source'
            )
            if hops > 0:
                S_km1, *_ = k_hop_subgraph(
                    sid, hops - 1, self.pyg_edge_index,
                    relabel_nodes=False,
                    num_nodes=self.num_entities,
                    flow='target_to_source'
                )
                exact = set(S_k.cpu().tolist()) - set(S_km1.cpu().tolist())
            else:
                exact = {sid}
            out.update(
                self.id_to_entity[i] for i in exact
                if self.id_to_entity[i] not in start_set
            )
        return list(out)

    # ---------- utilities ----------
    def get_available_methods(self) -> List[str]:
        return [
            'cugraph_khop',
            'dgl_khop',
            'pyg_khop',
        ]

    def get_graph_info(self) -> Dict:
        return {
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'num_triplets': len(self.triplets),
            'available_methods': self.get_available_methods(),
        }
