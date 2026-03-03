import networkx as nx
import igraph as ig
import snap
import graphblas as gb
import graph_tool.all as gt
from neo4j import GraphDatabase
import os
import math
import numpy as np
from typing import List, Tuple, Dict, Optional


class KnowledgeGraphRetrievalBaselinesCPU:

    def __init__(self, triplets: List[Tuple[str, str, str]], backend: Optional[str] = None):
        self.triplets = triplets
        self.backend = backend
        self._build_mappings()

        if backend == 'networkx':
            self._build_networkx_graph()
        elif backend == 'igraph':
            self._build_igraph_graph()
        elif backend == 'graphtool':
            self._build_graph_tool_graph()
        elif backend == 'snap':
            self._build_snap_graph()
        elif backend == 'graphblas':
            self._build_graphblas_graph()

        self.neo4j_driver = None

    def _build_mappings(self):
        entities = set()
        for s, r, o in self.triplets:
            entities.add(s)
            entities.add(o)
        self.idx_to_entity = sorted(list(entities))
        self.entity_to_id = {e: i for i, e in enumerate(self.idx_to_entity)}
        self.num_entities = len(self.idx_to_entity)

    # ---------- NetworkX ----------
    def _build_networkx_graph(self):
        if not hasattr(self, 'G'):
            self.G = nx.DiGraph()
            self.G.add_edges_from([(s, o) for s, r, o in self.triplets])

    def networkx_khop(self, start_cuis: List[str], hops: int) -> List[str]:
        if hops <= 0:
            return list(set(start_cuis))

        current = set(start_cuis)
        visited = set(start_cuis)

        for _ in range(hops):
            next_nodes = set()
            for node in current:
                if node in self.G:
                    next_nodes.update(self.G[node])

            next_nodes -= visited
            visited.update(next_nodes)
            current = next_nodes

        return list(current)

    # ---------- igraph ----------
    def _build_igraph_graph(self):
        if not hasattr(self, 'igraph_g'):
            edges = [(self.entity_to_id[s], self.entity_to_id[o])
                     for s, r, o in self.triplets]
            self.igraph_g = ig.Graph(
                n=self.num_entities, edges=edges, directed=True)

    def igraph_khop(self, start_cuis: List[str], hops: int) -> List[str]:
        if hops <= 0:
            return list(set(start_cuis))

        seed_ids = [self.entity_to_id[c]
                    for c in start_cuis if c in self.entity_to_id]
        if not seed_ids:
            return []

        ball_k = set().union(*self.igraph_g.neighborhood(vertices=seed_ids, order=hops, mode='OUT'))
        ball_k_minus_1 = set().union(
            *self.igraph_g.neighborhood(vertices=seed_ids, order=hops-1, mode='OUT'))

        frontier_ids = ball_k - ball_k_minus_1
        return [self.idx_to_entity[i] for i in frontier_ids]

    # ---------- Graph-Tool ----------
    def _build_graph_tool_graph(self):
        if not hasattr(self, 'graph_tool_g'):
            g = gt.Graph(directed=True)
            g.add_vertex(self.num_entities)
            edges = [(self.entity_to_id[s], self.entity_to_id[o])
                     for s, r, o in self.triplets]
            g.add_edge_list(edges)
            self.graph_tool_g = g

    def graphtool_khop(self, start_cuis, hops):
        if hops <= 0:
            return list(set(start_cuis))

        g = self.graph_tool_g
        seed_ids = [self.entity_to_id[c]
                    for c in start_cuis if c in self.entity_to_id]

        current = set(seed_ids)
        visited = set(seed_ids)

        for _ in range(hops):
            next_nodes = set()
            for v_idx in current:
                next_nodes.update(g.get_out_neighbors(v_idx))

            next_nodes -= visited
            visited.update(next_nodes)
            current = next_nodes

        return [self.idx_to_entity[i] for i in current]

    # ---------- SNAP ----------
    def _build_snap_graph(self):
        if not hasattr(self, 'snap_graph'):
            G = snap.TNGraph.New()
            for nid in range(self.num_entities):
                G.AddNode(nid)
            for s, r, o in self.triplets:
                G.AddEdge(self.entity_to_id[s], self.entity_to_id[o])
            self.snap_graph = G

    def snap_khop(self, start_cuis, hops):
        if hops <= 0:
            return list(set(start_cuis))

        current = {self.entity_to_id[c]
                   for c in start_cuis if c in self.entity_to_id}
        visited = set(current)

        for _ in range(hops):
            next_nodes = set()
            for nid in current:
                try:
                    NI = self.snap_graph.GetNI(nid)
                    for i in range(NI.GetOutDeg()):
                        next_nodes.add(NI.GetOutNId(i))
                except:
                    continue

            next_nodes -= visited
            visited.update(next_nodes)
            current = next_nodes

        return [self.idx_to_entity[i] for i in current]

    # ---------- GraphBLAS ----------
    def _build_graphblas_graph(self):
        if not hasattr(self, 'graphblas_matrix'):
            rows = [self.entity_to_id[s] for s, _, o in self.triplets]
            cols = [self.entity_to_id[o] for _, _, o in self.triplets]
            self.graphblas_matrix = gb.Matrix.from_coo(
                rows, cols, [True]*len(rows), nrows=self.num_entities, ncols=self.num_entities, dtype=bool, dup_op=gb.binary.lor
            )

    def graphblas_khop(self, start_cuis, hops):
        if hops <= 0:
            return list(set(start_cuis))

        seed_ids = [self.entity_to_id[c]
                    for c in start_cuis if c in self.entity_to_id]
        if not seed_ids:
            return []

        frontier = gb.Vector.from_coo(
            seed_ids, [True]*len(seed_ids), size=self.num_entities, dtype=bool)
        visited = frontier.dup()

        for _ in range(hops):
            frontier = frontier.vxm(
                self.graphblas_matrix, op=gb.semiring.lor_land).new(mask=~visited.S)
            visited = visited.ewise_add(frontier, op=gb.monoid.lor).new()

        idx, _ = frontier.to_coo()
        return [self.idx_to_entity[i] for i in idx]

    # ---------- Neo4j ----------
    def neo4j_khop(self, seeds: List[str], k: int):
        if not self.neo4j_driver:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    "bolt://localhost:7687", auth=("neo4j", "password"))
            except:
                return []

        query = """
        MATCH (s) WHERE s.id IN $seeds
        MATCH (s)-[:related_to*%d]->(t)
        WHERE NOT (s)-[:related_to*..%d]->(t)
        RETURN DISTINCT t.id as id
        """ % (k, k-1)

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(query, seeds=seeds)
                return [record["id"] for record in result]
        except:
            return []
