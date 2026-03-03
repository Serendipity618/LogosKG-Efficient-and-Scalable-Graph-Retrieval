"""
LogosKG Small: High-Performance Vectorized Knowledge Graph Engine

A production-grade library for efficient multi-hop knowledge graph retrieval.
Optimized for Retrieval-Augmented Generation (RAG) at scale.

Core Architecture:
    - Subject Matrix (Sub): CSR matrix mapping entities to triplet indices
    - Object Matrix (Obj): CSR matrix mapping triplet indices to entities  
    - Relation Matrix (Rel): CSR matrix mapping triplet indices to relations

Author: He Cheng, Yanjun Gao (LARK Lab at CU Anschutz)
License: MIT
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import scipy.sparse as sp
import torch
from numba import njit


class LogosKG:
    """High-performance knowledge graph retrieval engine with multiple backend support."""
    
    BACKEND_SCIPY = "scipy"
    BACKEND_NUMBA = "numba"
    BACKEND_TORCH = "torch"
    VALID_BACKENDS = frozenset([BACKEND_SCIPY, BACKEND_NUMBA, BACKEND_TORCH])
    
    def __init__(
        self, 
        triplets: List[Tuple[str, str, str]], 
        backend: str = "numba", 
        device: str = "cpu"
    ):
        """
        Initialize LogosKG engine.
        
        Args:
            triplets: List of (head, relation, tail) tuples
            backend: Computation backend ('scipy', 'numba', or 'torch')
            device: Device for torch backend ('cpu' or 'cuda')
        """
        if backend not in self.VALID_BACKENDS:
            raise ValueError(f"Invalid backend '{backend}'. Supported: {self.VALID_BACKENDS}")
        
        if backend == self.BACKEND_TORCH and device.startswith("cuda") and not torch.cuda.is_available():
            print("Warning: CUDA unavailable. Falling back to CPU.")
            device = "cpu"
        
        self.backend = backend
        self.device = device
        
        entities, relations = set(), set()
        for h, r, t in (triplets or []):
            entities.update([h, t])
            relations.add(r)
        
        self.entity_to_idx = {e: i for i, e in enumerate(sorted(entities))}
        self.idx_to_entity = {i: e for e, i in self.entity_to_idx.items()}
        self.relation_to_idx = {r: i for i, r in enumerate(sorted(relations))}
        self.idx_to_relation = {i: r for r, i in self.relation_to_idx.items()}
        self.num_entities = len(entities)
        self.num_relations = len(relations)
        
        self._build_topology(triplets)
        
        self._hop_kernel = getattr(self, f"_hop_{backend}")
        self._expand_kernel = getattr(self, f"_expand_{backend}")

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def retrieve_at_k_hop(
        self, 
        entity_ids: List[str], 
        hops: int, 
        shortest_path: bool = True
    ) -> List[str]:
        """Retrieve entities exactly at K hops from seeds."""
        if hops < 0:
            raise ValueError("Hops cannot be negative.")
        
        valid_seeds = [e for e in entity_ids if e in self.entity_to_idx]
        if not valid_seeds:
            return []
        if hops == 0:
            return list(set(valid_seeds))
        
        if self.backend == self.BACKEND_TORCH and len(valid_seeds) > 1 and self.device.startswith("cuda"):
            return self._retrieve_at_k_hop_torch_batched(valid_seeds, hops, shortest_path)
        
        indices = [self.entity_to_idx[e] for e in valid_seeds]
        frontier = np.zeros(self.num_entities, dtype=bool)
        frontier[indices] = True
        
        if self.backend == self.BACKEND_TORCH:
            frontier = torch.from_numpy(frontier).to(self.device).bool()
            visited = frontier.clone() if shortest_path else None
        else:
            visited = frontier.copy() if shortest_path else None
        
        for _ in range(hops):
            frontier = self._hop_kernel(frontier)
            
            if shortest_path:
                if self.backend == self.BACKEND_TORCH:
                    frontier = (frontier > 0) & ~visited
                    if not frontier.any():
                        return []
                    visited |= frontier
                else:
                    frontier &= ~visited
                    if not frontier.any():
                        return []
                    visited |= frontier
        
        if isinstance(frontier, torch.Tensor):
            indices = torch.nonzero(frontier).flatten().cpu().numpy()
        else:
            indices = np.where(frontier)[0]
        
        return [self.idx_to_entity[i] for i in indices]

    def retrieve_within_k_hop(
        self, 
        entity_ids: List[str], 
        hops: int, 
        shortest_path: bool = True
    ) -> List[str]:
        """Retrieve all entities within K hops from seeds."""
        if hops < 0:
            raise ValueError("Hops cannot be negative.")
        
        valid_seeds = [e for e in entity_ids if e in self.entity_to_idx]
        if not valid_seeds:
            return []
        if hops == 0:
            return list(set(valid_seeds))
        
        if self.backend == self.BACKEND_TORCH and len(valid_seeds) > 1 and self.device.startswith("cuda"):
            return self._retrieve_within_k_hop_torch_batched(valid_seeds, hops, shortest_path)
        
        indices = [self.entity_to_idx[e] for e in valid_seeds]
        frontier = np.zeros(self.num_entities, dtype=bool)
        frontier[indices] = True
        
        if self.backend == self.BACKEND_TORCH:
            frontier = torch.from_numpy(frontier).to(self.device).bool()
            accumulated = frontier.clone()
            visited = frontier.clone() if shortest_path else None
        else:
            accumulated = frontier.copy()
            visited = frontier.copy() if shortest_path else None
        
        for _ in range(hops):
            frontier = self._hop_kernel(frontier)
            
            if self.backend == self.BACKEND_TORCH:
                if shortest_path:
                    frontier = (frontier > 0) & ~visited
                if not frontier.any():
                    break
                if shortest_path:
                    visited |= frontier
                accumulated |= frontier
            else:
                if shortest_path:
                    frontier &= ~visited
                if not frontier.any():
                    break
                if shortest_path:
                    visited |= frontier
                accumulated |= frontier
        
        if isinstance(accumulated, torch.Tensor):
            indices = torch.nonzero(accumulated).flatten().cpu().numpy()
        else:
            indices = np.where(accumulated)[0]
        
        return [self.idx_to_entity[i] for i in indices]

    def retrieve_with_paths_at_k_hop(
        self, 
        entity_ids: List[str], 
        hops: int = 2, 
        shortest_path: bool = True,
        max_paths_per_entity: Optional[int] = None
    ) -> Dict[str, Any]:
        """Retrieve entities and paths at exactly K hops."""
        if hops < 0:
            raise ValueError("Hops cannot be negative.")
        
        valid_seeds = [e for e in entity_ids if e in self.entity_to_idx]
        if not valid_seeds:
            return {"entities": [], "paths": {}}
        if hops == 0:
            return {"entities": valid_seeds, "paths": {e: [[e]] for e in valid_seeds}}
        
        if self.backend == self.BACKEND_TORCH and len(valid_seeds) > 1 and self.device.startswith("cuda"):
            return self._retrieve_with_paths_at_k_hop_torch_batched(valid_seeds, hops, shortest_path, max_paths_per_entity)
        
        seed_indices = [self.entity_to_idx[s] for s in valid_seeds]
        
        if self.backend == self.BACKEND_TORCH:
            paths = torch.tensor([[s] for s in seed_indices], dtype=torch.long, device=self.device)
            visited = torch.zeros(self.num_entities, dtype=torch.bool, device=self.device) if shortest_path else None
        else:
            paths = np.array([[s] for s in seed_indices], dtype=np.int32)
            visited = np.zeros(self.num_entities, dtype=bool) if shortest_path else None
        
        if visited is not None:
            visited[seed_indices] = True
        
        for _ in range(hops):
            paths = self._expand_kernel(paths, visited)
            if len(paths) == 0:
                return {"entities": [], "paths": {}}
            if visited is not None:
                visited[paths[:, -1]] = True
        
        return self._decode_paths(paths, max_paths_per_entity)

    def retrieve_with_paths_within_k_hop(
        self, 
        entity_ids: List[str], 
        hops: int = 2, 
        shortest_path: bool = True, 
        max_paths_per_entity: Optional[int] = None
    ) -> Dict[str, Any]:
        """Retrieve all entities and paths within K hops."""
        if hops < 0:
            raise ValueError("Hops cannot be negative.")
        
        valid_seeds = [e for e in entity_ids if e in self.entity_to_idx]
        if not valid_seeds:
            return {"entities": [], "paths": {}}
        if hops == 0:
            return {"entities": valid_seeds, "paths": {e: [[e]] for e in valid_seeds}}
        
        if self.backend == self.BACKEND_TORCH and len(valid_seeds) > 1 and self.device.startswith("cuda"):
            return self._retrieve_with_paths_within_k_hop_torch_batched(valid_seeds, hops, shortest_path, max_paths_per_entity)
        
        seed_indices = [self.entity_to_idx[s] for s in valid_seeds]
        
        if self.backend == self.BACKEND_TORCH:
            paths = torch.tensor([[s] for s in seed_indices], dtype=torch.long, device=self.device)
            visited = torch.zeros(self.num_entities, dtype=torch.bool, device=self.device) if shortest_path else None
        else:
            paths = np.array([[s] for s in seed_indices], dtype=np.int32)
            visited = np.zeros(self.num_entities, dtype=bool) if shortest_path else None
        
        if visited is not None:
            visited[seed_indices] = True
        
        all_paths = [paths]
        
        for _ in range(hops):
            paths = self._expand_kernel(paths, visited)
            if len(paths) == 0:
                break
            if visited is not None:
                visited[paths[:, -1]] = True
            all_paths.append(paths)
        
        final_results = {}
        for path_group in all_paths:
            decoded = self._decode_paths(path_group, max_paths_per_entity)
            for entity, path_list in decoded["paths"].items():
                if entity not in final_results:
                    final_results[entity] = []
                final_results[entity].extend(path_list)
        
        return {"entities": list(final_results.keys()), "paths": final_results}

    # =========================================================================
    # GPU BATCHING OPTIMIZATION
    # =========================================================================

    def _retrieve_at_k_hop_torch_batched(self, valid_seeds: List[str], hops: int, shortest_path: bool) -> List[str]:
        """GPU-optimized batched retrieval for multiple seeds."""
        seed_indices = [self.entity_to_idx[e] for e in valid_seeds]
        n_seeds = len(seed_indices)
        
        current = torch.zeros((n_seeds, self.num_entities), dtype=torch.bool, device=self.device)
        for i, idx in enumerate(seed_indices):
            current[i, idx] = True
        
        if shortest_path:
            visited = current.clone()
        
        for _ in range(hops):
            trips = torch.sparse.mm(current.float(), self.sub_matrix)
            next_ents = torch.sparse.mm(trips, self.obj_matrix)
            current = next_ents > 0
            
            if shortest_path:
                current = current & ~visited
                if not current.any():
                    break
                visited |= current
        
        final = current.any(dim=0)
        for idx in seed_indices:
            final[idx] = False
        
        indices = torch.nonzero(final).flatten().cpu().numpy()
        return [self.idx_to_entity[i] for i in indices]

    def _retrieve_within_k_hop_torch_batched(self, valid_seeds: List[str], hops: int, shortest_path: bool) -> List[str]:
        """GPU-optimized batched retrieval within K hops."""
        seed_indices = [self.entity_to_idx[e] for e in valid_seeds]
        n_seeds = len(seed_indices)
        
        current = torch.zeros((n_seeds, self.num_entities), dtype=torch.bool, device=self.device)
        for i, idx in enumerate(seed_indices):
            current[i, idx] = True
        
        accumulated = current.clone()
        if shortest_path:
            visited = current.clone()
        
        for _ in range(hops):
            trips = torch.sparse.mm(current.float(), self.sub_matrix)
            next_ents = torch.sparse.mm(trips, self.obj_matrix)
            current = next_ents > 0
            
            if shortest_path:
                current = current & ~visited
                if not current.any():
                    break
                visited |= current
            
            accumulated |= current
        
        final = accumulated.any(dim=0)
        for idx in seed_indices:
            final[idx] = False
        
        indices = torch.nonzero(final).flatten().cpu().numpy()
        return [self.idx_to_entity[i] for i in indices]

    def _retrieve_with_paths_at_k_hop_torch_batched(self, valid_seeds: List[str], hops: int, shortest_path: bool, max_paths: Optional[int]) -> Dict[str, Any]:
        """GPU-optimized batched path retrieval at K hops."""
        all_results = {"entities": [], "paths": {}}
        
        for seed in valid_seeds:
            result = self.retrieve_with_paths_at_k_hop([seed], hops, shortest_path, max_paths)
            all_results["entities"].extend(result["entities"])
            for entity, paths in result["paths"].items():
                if entity not in all_results["paths"]:
                    all_results["paths"][entity] = []
                all_results["paths"][entity].extend(paths)
        
        all_results["entities"] = list(set(all_results["entities"]))
        return all_results

    def _retrieve_with_paths_within_k_hop_torch_batched(self, valid_seeds: List[str], hops: int, shortest_path: bool, max_paths: Optional[int]) -> Dict[str, Any]:
        """GPU-optimized batched path retrieval within K hops."""
        all_results = {"entities": [], "paths": {}}
        
        for seed in valid_seeds:
            result = self.retrieve_with_paths_within_k_hop([seed], hops, shortest_path, max_paths)
            all_results["entities"].extend(result["entities"])
            for entity, paths in result["paths"].items():
                if entity not in all_results["paths"]:
                    all_results["paths"][entity] = []
                all_results["paths"][entity].extend(paths)
        
        all_results["entities"] = list(set(all_results["entities"]))
        return all_results

    # =========================================================================
    # BACKEND KERNELS
    # =========================================================================

    def _hop_scipy(self, frontier: np.ndarray) -> np.ndarray:
        """SciPy: frontier @ Sub @ Obj"""
        vec = sp.csr_matrix(frontier.reshape(1, -1))
        result = (vec @ self.sub_matrix @ self.obj_matrix).toarray().ravel()
        return result > 0

    def _hop_numba(self, frontier: np.ndarray) -> np.ndarray:
        """Numba: Direct CSR traversal"""
        active = np.where(frontier)[0].astype(np.int32)
        result = _numba_hop_kernel(active, self.sub_indptr, self.sub_indices, self.obj_indices, self.num_entities)
        return result.astype(bool)

    def _hop_torch(self, frontier: torch.Tensor) -> torch.Tensor:
        """PyTorch: frontier @ Sub @ Obj"""
        active = torch.nonzero(frontier).flatten()
        if active.numel() == 0:
            return torch.zeros_like(frontier)
        
        sub_crow = self.sub_matrix.crow_indices()
        sub_col = self.sub_matrix.col_indices()
        
        starts = sub_crow[active]
        ends = sub_crow[active + 1]
        counts = ends - starts
        
        if counts.sum() == 0:
            return torch.zeros_like(frontier)
        
        row_ids = torch.repeat_interleave(torch.arange(len(starts), device=self.device), counts)
        cumsum = torch.cat([torch.zeros(1, device=self.device, dtype=torch.long), counts[:-1].cumsum(0)])
        offsets = torch.arange(counts.sum().item(), device=self.device) - torch.repeat_interleave(cumsum, counts)
        triplet_indices = sub_col[starts[row_ids] + offsets]
        
        obj_col = self.obj_matrix.col_indices()
        tail_entities = obj_col[triplet_indices]
        
        result = torch.zeros_like(frontier)
        result[tail_entities] = True
        return result

    def _expand_scipy(self, paths: np.ndarray, visited: Optional[np.ndarray]) -> np.ndarray:
        """SciPy: Expand paths using Sub, Rel, Obj matrices"""
        last_nodes = paths[:, -1]
        
        sub_slice = self.sub_matrix[last_nodes]
        triplet_indices = sub_slice.indices
        
        parents = np.repeat(np.arange(len(last_nodes)), np.diff(sub_slice.indptr))
        
        rel_slice = self.rel_matrix[triplet_indices]
        relations = rel_slice.indices
        
        obj_slice = self.obj_matrix[triplet_indices]
        tails = obj_slice.indices
        
        valid = np.ones(len(tails), dtype=bool)
        if visited is not None:
            valid &= ~visited[tails]
        
        if not valid.any():
            return np.array([])
        
        return np.hstack([paths[parents[valid]], relations[valid].reshape(-1, 1), tails[valid].reshape(-1, 1)])

    def _expand_numba(self, paths: np.ndarray, visited: Optional[np.ndarray]) -> np.ndarray:
        """Numba: Expand paths using CSR arrays"""
        last_nodes = paths[:, -1]
        parents, triplet_indices = _numba_expand_kernel(last_nodes, self.sub_indptr, self.sub_indices)
        
        relations = self.rel_indices[triplet_indices]
        tails = self.obj_indices[triplet_indices]
        
        valid = np.ones(len(tails), dtype=bool)
        if visited is not None:
            valid &= ~visited[tails]
        
        if not valid.any():
            return np.array([])
        
        return np.hstack([paths[parents[valid]], relations[valid].reshape(-1, 1), tails[valid].reshape(-1, 1)])

    def _expand_torch(self, paths: torch.Tensor, visited: Optional[torch.Tensor]) -> torch.Tensor:
        """PyTorch: Expand paths using Sub, Rel, Obj matrices"""
        last_nodes = paths[:, -1]
        
        sub_crow = self.sub_matrix.crow_indices()
        sub_col = self.sub_matrix.col_indices()
        
        starts, ends = sub_crow[last_nodes], sub_crow[last_nodes + 1]
        counts = ends - starts
        total = counts.sum().item()
        
        if total == 0:
            return torch.tensor([], dtype=torch.long, device=self.device).reshape(0, paths.shape[1])
        
        parents = torch.repeat_interleave(torch.arange(len(last_nodes), device=self.device), counts)
        cumsum = torch.cat([torch.zeros(1, device=self.device, dtype=torch.long), counts[:-1].cumsum(0)])
        offsets = torch.arange(total, device=self.device) - torch.repeat_interleave(cumsum, counts)
        
        row_expanded = torch.repeat_interleave(torch.arange(len(starts), device=self.device), counts)
        triplet_indices = sub_col[starts[row_expanded] + offsets]
        
        rel_col = self.rel_matrix.col_indices()
        relations = rel_col[triplet_indices]
        
        obj_col = self.obj_matrix.col_indices()
        tails = obj_col[triplet_indices]
        
        valid = torch.ones(len(tails), dtype=torch.bool, device=self.device)
        if visited is not None:
            valid &= ~visited[tails]
        
        if not valid.any():
            return torch.tensor([], dtype=torch.long, device=self.device)
        
        return torch.cat([paths[parents[valid]], relations[valid].unsqueeze(1), tails[valid].unsqueeze(1)], dim=1)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _decode_paths(self, paths: Union[np.ndarray, torch.Tensor], max_paths: Optional[int]) -> Dict[str, Any]:
        """Decode path tensor to human-readable format."""
        if isinstance(paths, torch.Tensor):
            paths = paths.cpu().numpy()
        
        results = {}
        for path in paths:
            end = self.idx_to_entity[path[-1]]
            if end not in results:
                results[end] = []
            if max_paths and len(results[end]) >= max_paths:
                continue
            
            if len(path) == 1:
                decoded = [self.idx_to_entity[path[0]]]
            else:
                decoded = [self.idx_to_entity[path[0]]]
                for i in range(1, len(path), 2):
                    decoded.extend([f"--{self.idx_to_relation[path[i]]}-->", self.idx_to_entity[path[i + 1]]])
            
            results[end].append(decoded)
        
        return {"entities": list(results.keys()), "paths": results}

    def _build_topology(self, triplets: List[Tuple[str, str, str]]) -> None:
        """Build Sub, Rel, Obj sparse matrices in CSR format."""
        if not triplets:
            return
        
        encoded = [(self.entity_to_idx[h], self.relation_to_idx[r], self.entity_to_idx[t]) for h, r, t in triplets]
        heads = np.array([h for h, r, t in encoded], dtype=np.int32)
        relations = np.array([r for h, r, t in encoded], dtype=np.int32)
        tails = np.array([t for h, r, t in encoded], dtype=np.int32)
        num_triplets = len(encoded)
        
        if self.backend == self.BACKEND_SCIPY:
            self.sub_matrix = sp.csr_matrix(
                (np.ones(num_triplets, bool), (heads, np.arange(num_triplets))),
                shape=(self.num_entities, num_triplets)
            ).sorted_indices()
            
            self.rel_matrix = sp.csr_matrix(
                (np.ones(num_triplets, bool), (np.arange(num_triplets), relations)),
                shape=(num_triplets, self.num_relations)
            )
            
            self.obj_matrix = sp.csr_matrix(
                (np.ones(num_triplets, bool), (np.arange(num_triplets), tails)),
                shape=(num_triplets, self.num_entities)
            )
        
        elif self.backend == self.BACKEND_NUMBA:
            order = np.argsort(heads)
            sorted_heads = heads[order]
            sorted_relations = relations[order]
            sorted_tails = tails[order]
            
            counts = np.bincount(sorted_heads, minlength=self.num_entities)
            self.sub_indptr = np.zeros(self.num_entities + 1, dtype=np.int32)
            self.sub_indptr[1:] = np.cumsum(counts)
            self.sub_indices = np.arange(num_triplets, dtype=np.int32)
            
            self.rel_indices = sorted_relations.astype(np.int32)
            self.obj_indices = sorted_tails.astype(np.int32)
        
        elif self.backend == self.BACKEND_TORCH:
            dev = self.device
            
            order = np.argsort(heads)
            sorted_heads = heads[order]
            sorted_relations = relations[order]
            sorted_tails = tails[order]
            
            counts = np.bincount(sorted_heads, minlength=self.num_entities)
            sub_crow = np.zeros(self.num_entities + 1, dtype=np.int64)
            sub_crow[1:] = np.cumsum(counts)
            
            self.sub_matrix = torch.sparse_csr_tensor(
                crow_indices=torch.from_numpy(sub_crow).to(dev),
                col_indices=torch.arange(num_triplets, device=dev, dtype=torch.long),
                values=torch.ones(num_triplets, device=dev),
                size=(self.num_entities, num_triplets),
                device=dev
            )
            
            rel_crow = torch.arange(num_triplets + 1, device=dev, dtype=torch.long)
            
            self.rel_matrix = torch.sparse_csr_tensor(
                crow_indices=rel_crow,
                col_indices=torch.from_numpy(sorted_relations).to(dev).long(),
                values=torch.ones(num_triplets, device=dev),
                size=(num_triplets, self.num_relations),
                device=dev
            )
            
            obj_crow = torch.arange(num_triplets + 1, device=dev, dtype=torch.long)
            
            self.obj_matrix = torch.sparse_csr_tensor(
                crow_indices=obj_crow,
                col_indices=torch.from_numpy(sorted_tails).to(dev).long(),
                values=torch.ones(num_triplets, device=dev),
                size=(num_triplets, self.num_entities),
                device=dev
            )


# =========================================================================
# NUMBA KERNELS
# =========================================================================

@njit(cache=True, fastmath=True, boundscheck=False)
def _numba_expand_kernel(last_nodes, sub_indptr, sub_indices):
    """Expand paths using Sub matrix CSR structure."""
    num_nodes = len(last_nodes)
    output_size = 0
    edge_counts = np.empty(num_nodes, dtype=np.int32)
    
    for i in range(num_nodes):
        node = last_nodes[i]
        count = sub_indptr[node + 1] - sub_indptr[node]
        edge_counts[i] = count
        output_size += count
    
    parents = np.empty(output_size, dtype=np.int32)
    triplet_indices = np.empty(output_size, dtype=np.int32)
    
    cursor = 0
    for i in range(num_nodes):
        count = edge_counts[i]
        if count > 0:
            node = last_nodes[i]
            start = sub_indptr[node]
            parents[cursor:cursor + count] = i
            for offset in range(count):
                triplet_indices[cursor + offset] = sub_indices[start + offset]
            cursor += count
    
    return parents, triplet_indices


@njit(cache=True, fastmath=True)
def _numba_hop_kernel(active_indices, sub_indptr, sub_indices, obj_indices, num_entities):
    """One-hop expansion: active @ Sub @ Obj."""
    result = np.zeros(num_entities, dtype=np.uint8)
    for i in range(len(active_indices)):
        entity = active_indices[i]
        for csr_pos in range(sub_indptr[entity], sub_indptr[entity + 1]):
            triplet_id = sub_indices[csr_pos]
            result[obj_indices[triplet_id]] = 1
    return result