"""
LogosKG Large: Scalable Partitioned Knowledge Graph Engine

A disk-backed partitioned engine for massive knowledge graphs that exceed memory capacity.
Employs LRU caching to load partitions on-demand while maintaining full consistency with
LogosKG Small's API, algorithms, and optimizations.

Core Architecture:
    - Subject Matrix (Sub): CSR matrix mapping entities to triplet indices
    - Object Matrix (Obj): CSR matrix mapping triplet indices to entities  
    - Relation Matrix (Rel): CSR matrix mapping triplet indices to relations
    - Partitioning: Graph split into disk-backed partitions
    - LRU Cache: Memory-efficient partition management

Author: He Cheng, Yanjun Gao (LARK Lab at CU Anschutz)
License: MIT
Version: 1.0.0
"""

import os
import pickle
import tempfile
import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict, OrderedDict
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from numba import njit
from LogosKG.utils.KGPartitioner import KnowledgeGraphPartitioner


class LogosKGLarge:
    """Scalable knowledge graph retrieval engine with disk-backed partitioning."""

    BACKEND_SCIPY = "scipy"
    BACKEND_NUMBA = "numba"
    BACKEND_TORCH = "torch"
    VALID_BACKENDS = frozenset([BACKEND_SCIPY, BACKEND_NUMBA, BACKEND_TORCH])

    def __init__(
        self,
        partition_dir: str,
        backend: str = "numba",
        device: str = "cpu",
        cache_size: int = 10,
        triplets_for_auto: Optional[List[Tuple[str, str, str]]] = None,
        num_partitions: int = 16
    ):
        """
        Initialize LogosKG Large engine.

        Args:
            partition_dir: Directory containing partitioned graph data
            backend: Computation backend ('scipy', 'numba', or 'torch')
            device: Device for torch backend ('cpu' or 'cuda')
            cache_size: Number of partitions to keep in memory
            triplets_for_auto: Triplets for auto-partitioning if partitions don't exist
            num_partitions: Number of partitions to create (if auto-partitioning)
        """
        if backend not in self.VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend '{backend}'. Supported: {self.VALID_BACKENDS}")

        if backend == self.BACKEND_TORCH and device.startswith("cuda") and not torch.cuda.is_available():
            print("Warning: CUDA unavailable. Falling back to CPU.")
            device = "cpu"

        self.partition_dir = partition_dir
        self.backend = backend
        self.device = device
        self.cache_size = cache_size
        self.num_partitions = num_partitions

        if not self._check_partitions_exist():
            if triplets_for_auto is None:
                raise ValueError(
                    f"Partitions not found in '{partition_dir}' and no triplets provided for auto-partitioning."
                )
            print(
                f"Partitions not found. Auto-partitioning graph into {num_partitions} partitions...")
            self._auto_partition(triplets_for_auto)

        self._load_metadata()

        self.partition_cache = OrderedDict()

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
            frontier = self._hop_across_partitions(frontier)

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
            frontier = self._hop_across_partitions(frontier)

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
            paths = torch.tensor([[s] for s in seed_indices],
                                 dtype=torch.long, device=self.device)
            visited = torch.zeros(
                self.num_entities, dtype=torch.bool, device=self.device) if shortest_path else None
        else:
            paths = np.array([[s] for s in seed_indices], dtype=np.int32)
            visited = np.zeros(self.num_entities,
                               dtype=bool) if shortest_path else None

        if visited is not None:
            visited[seed_indices] = True

        for _ in range(hops):
            paths = self._expand_paths_across_partitions(paths, visited)
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
            paths = torch.tensor([[s] for s in seed_indices],
                                 dtype=torch.long, device=self.device)
            visited = torch.zeros(
                self.num_entities, dtype=torch.bool, device=self.device) if shortest_path else None
        else:
            paths = np.array([[s] for s in seed_indices], dtype=np.int32)
            visited = np.zeros(self.num_entities,
                               dtype=bool) if shortest_path else None

        if visited is not None:
            visited[seed_indices] = True

        all_paths = [paths]

        for _ in range(hops):
            paths = self._expand_paths_across_partitions(paths, visited)
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

        current = torch.zeros(
            (n_seeds, self.num_entities), dtype=torch.bool, device=self.device)
        for i, idx in enumerate(seed_indices):
            current[i, idx] = True

        if shortest_path:
            visited = current.clone()

        for _ in range(hops):
            next_current = torch.zeros_like(current)

            for seed_idx in range(n_seeds):
                active = torch.nonzero(
                    current[seed_idx]).flatten().cpu().numpy()
                if len(active) == 0:
                    continue

                partition_groups = defaultdict(list)
                for entity_idx in active:
                    partition_id = self.partition_map[entity_idx]
                    partition_groups[partition_id].append(entity_idx)

                for partition_id, entities in partition_groups.items():
                    partition = self._get_partition(partition_id)
                    entities_t = torch.tensor(
                        entities, dtype=torch.long, device=self.device)

                    sub_crow = partition["sub_matrix"].crow_indices()
                    sub_col = partition["sub_matrix"].col_indices()

                    starts = sub_crow[entities_t]
                    ends = sub_crow[entities_t + 1]
                    counts = ends - starts

                    if counts.sum() == 0:
                        continue

                    row_ids = torch.repeat_interleave(torch.arange(
                        len(starts), device=self.device), counts)
                    cumsum = torch.cat(
                        [torch.zeros(1, device=self.device, dtype=torch.long), counts[:-1].cumsum(0)])
                    offsets = torch.arange(counts.sum().item(
                    ), device=self.device) - torch.repeat_interleave(cumsum, counts)
                    triplet_indices = sub_col[starts[row_ids] + offsets]

                    obj_col = partition["obj_matrix"].col_indices()
                    tail_entities = obj_col[triplet_indices]

                    next_current[seed_idx, tail_entities] = True

            current = next_current

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

        current = torch.zeros(
            (n_seeds, self.num_entities), dtype=torch.bool, device=self.device)
        for i, idx in enumerate(seed_indices):
            current[i, idx] = True

        accumulated = current.clone()
        if shortest_path:
            visited = current.clone()

        for _ in range(hops):
            next_current = torch.zeros_like(current)

            for seed_idx in range(n_seeds):
                active = torch.nonzero(
                    current[seed_idx]).flatten().cpu().numpy()
                if len(active) == 0:
                    continue

                partition_groups = defaultdict(list)
                for entity_idx in active:
                    partition_id = self.partition_map[entity_idx]
                    partition_groups[partition_id].append(entity_idx)

                for partition_id, entities in partition_groups.items():
                    partition = self._get_partition(partition_id)
                    entities_t = torch.tensor(
                        entities, dtype=torch.long, device=self.device)

                    sub_crow = partition["sub_matrix"].crow_indices()
                    sub_col = partition["sub_matrix"].col_indices()

                    starts = sub_crow[entities_t]
                    ends = sub_crow[entities_t + 1]
                    counts = ends - starts

                    if counts.sum() == 0:
                        continue

                    row_ids = torch.repeat_interleave(torch.arange(
                        len(starts), device=self.device), counts)
                    cumsum = torch.cat(
                        [torch.zeros(1, device=self.device, dtype=torch.long), counts[:-1].cumsum(0)])
                    offsets = torch.arange(counts.sum().item(
                    ), device=self.device) - torch.repeat_interleave(cumsum, counts)
                    triplet_indices = sub_col[starts[row_ids] + offsets]

                    obj_col = partition["obj_matrix"].col_indices()
                    tail_entities = obj_col[triplet_indices]

                    next_current[seed_idx, tail_entities] = True

            current = next_current

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
            result = self.retrieve_with_paths_at_k_hop(
                [seed], hops, shortest_path, max_paths)
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
            result = self.retrieve_with_paths_within_k_hop(
                [seed], hops, shortest_path, max_paths)
            all_results["entities"].extend(result["entities"])
            for entity, paths in result["paths"].items():
                if entity not in all_results["paths"]:
                    all_results["paths"][entity] = []
                all_results["paths"][entity].extend(paths)

        all_results["entities"] = list(set(all_results["entities"]))
        return all_results

    # =========================================================================
    # PARTITION MANAGEMENT
    # =========================================================================

    def _check_partitions_exist(self) -> bool:
        """Check if partition directory contains required metadata."""
        metadata_path = os.path.join(self.partition_dir, "metadata.pkl")
        old_metadata_path = os.path.join(
            self.partition_dir, "subject_to_partition.pkl")
        return os.path.exists(metadata_path) or os.path.exists(old_metadata_path)

    def _auto_partition(self, triplets: List[Tuple[str, str, str]]) -> None:
        """Auto-partition the graph if partitions don't exist."""

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            temp_path = f.name
            for head, relation, tail in triplets:
                f.write(f"{head}|{relation}|{tail}\n")

        partitioner = KnowledgeGraphPartitioner(
            input_path=temp_path,
            output_dir=self.partition_dir,
            num_partitions=self.num_partitions,
            input_type="triplets"
        )
        partitioner.partition()

        if os.path.exists(temp_path):
            os.remove(temp_path)

        self._build_metadata_from_old_format()

        print(
            f"Partitioning complete. {self.num_partitions} partitions created.")

    def _build_metadata_from_old_format(self) -> None:
        """Convert OLD partitioner output to metadata.pkl format."""
        subject_to_partition_path = os.path.join(
            self.partition_dir, "subject_to_partition.pkl")
        with open(subject_to_partition_path, "rb") as f:
            subject_to_partition = pickle.load(f)

        entities = set()
        relations = set()

        for partition_id in range(self.num_partitions):
            partition_path = os.path.join(
                self.partition_dir, f"part_{partition_id}.pkl")
            if os.path.exists(partition_path):
                with open(partition_path, "rb") as f:
                    triplets = pickle.load(f)
                    for subject, relation, obj in triplets:
                        entities.add(subject)
                        entities.add(obj)
                        relations.add(relation)

        entity_to_idx = {e: i for i, e in enumerate(sorted(entities))}
        idx_to_entity = {i: e for e, i in entity_to_idx.items()}
        relation_to_idx = {r: i for i, r in enumerate(sorted(relations))}
        idx_to_relation = {i: r for r, i in relation_to_idx.items()}

        partition_map = {}
        for subject, partition_id in subject_to_partition.items():
            if subject in entity_to_idx:
                partition_map[entity_to_idx[subject]] = partition_id

        metadata = {
            "entity_to_idx": entity_to_idx,
            "idx_to_entity": idx_to_entity,
            "relation_to_idx": relation_to_idx,
            "idx_to_relation": idx_to_relation,
            "num_entities": len(entity_to_idx),
            "num_relations": len(relation_to_idx),
            "num_partitions": self.num_partitions,
            "partition_map": partition_map
        }

        metadata_path = os.path.join(self.partition_dir, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_metadata(self) -> None:
        """Load global entity/relation mappings and partition index."""
        metadata_path = os.path.join(self.partition_dir, "metadata.pkl")

        if not os.path.exists(metadata_path):
            self._build_metadata_from_old_format()

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        self.entity_to_idx = metadata["entity_to_idx"]
        self.idx_to_entity = metadata["idx_to_entity"]
        self.relation_to_idx = metadata["relation_to_idx"]
        self.idx_to_relation = metadata["idx_to_relation"]
        self.num_entities = metadata["num_entities"]
        self.num_relations = metadata["num_relations"]
        self.partition_map = metadata["partition_map"]

    def _get_partition(self, partition_id: int) -> Dict[str, Any]:
        """Retrieve partition from LRU cache or load from disk."""
        if partition_id in self.partition_cache:
            self.partition_cache.move_to_end(partition_id)
            return self.partition_cache[partition_id]

        new_format_path = os.path.join(
            self.partition_dir, f"partition_{partition_id}.pkl")
        old_format_path = os.path.join(
            self.partition_dir, f"part_{partition_id}.pkl")

        if os.path.exists(new_format_path):
            with open(new_format_path, "rb") as f:
                raw_partition = pickle.load(f)
        elif os.path.exists(old_format_path):
            with open(old_format_path, "rb") as f:
                raw_partition = pickle.load(f)
        else:
            raise FileNotFoundError(f"Partition {partition_id} not found")

        partition = self._build_partition_matrices(raw_partition)

        if len(self.partition_cache) >= self.cache_size:
            self.partition_cache.popitem(last=False)

        self.partition_cache[partition_id] = partition
        return partition

    def _build_partition_matrices(self, raw_partition: Union[List, Dict[str, Any]]) -> Dict[str, Any]:
        """Build Sub, Rel, Obj matrices for a partition."""
        if isinstance(raw_partition, list):
            heads = []
            relations = []
            tails = []

            for subject, relation, obj in raw_partition:
                heads.append(self.entity_to_idx[subject])
                relations.append(self.relation_to_idx[relation])
                tails.append(self.entity_to_idx[obj])

            heads = np.array(heads, dtype=np.int32)
            relations = np.array(relations, dtype=np.int32)
            tails = np.array(tails, dtype=np.int32)
        else:
            heads = np.array(raw_partition["head_indices"], dtype=np.int32)
            relations = np.array(
                raw_partition["relation_indices"], dtype=np.int32)
            tails = np.array(raw_partition["tail_indices"], dtype=np.int32)

        num_triplets = len(heads)
        partition = {}

        if self.backend == self.BACKEND_SCIPY:
            partition["sub_matrix"] = sp.csr_matrix(
                (np.ones(num_triplets, bool), (heads, np.arange(num_triplets))),
                shape=(self.num_entities, num_triplets)
            ).sorted_indices()

            partition["rel_matrix"] = sp.csr_matrix(
                (np.ones(num_triplets, bool), (np.arange(num_triplets), relations)),
                shape=(num_triplets, self.num_relations)
            )

            partition["obj_matrix"] = sp.csr_matrix(
                (np.ones(num_triplets, bool), (np.arange(num_triplets), tails)),
                shape=(num_triplets, self.num_entities)
            )

        elif self.backend == self.BACKEND_NUMBA:
            order = np.argsort(heads)
            sorted_heads = heads[order]
            sorted_relations = relations[order]
            sorted_tails = tails[order]

            counts = np.bincount(sorted_heads, minlength=self.num_entities)
            sub_indptr = np.zeros(self.num_entities + 1, dtype=np.int32)
            sub_indptr[1:] = np.cumsum(counts)
            sub_indices = np.arange(num_triplets, dtype=np.int32)

            partition["sub_indptr"] = sub_indptr
            partition["sub_indices"] = sub_indices
            partition["rel_indices"] = sorted_relations.astype(np.int32)
            partition["obj_indices"] = sorted_tails.astype(np.int32)

        elif self.backend == self.BACKEND_TORCH:
            dev = self.device

            order = np.argsort(heads)
            sorted_heads = heads[order]
            sorted_relations = relations[order]
            sorted_tails = tails[order]

            counts = np.bincount(sorted_heads, minlength=self.num_entities)
            sub_crow = np.zeros(self.num_entities + 1, dtype=np.int64)
            sub_crow[1:] = np.cumsum(counts)

            partition["sub_matrix"] = torch.sparse_csr_tensor(
                crow_indices=torch.from_numpy(sub_crow).to(dev),
                col_indices=torch.arange(
                    num_triplets, device=dev, dtype=torch.long),
                values=torch.ones(num_triplets, device=dev),
                size=(self.num_entities, num_triplets),
                device=dev
            )

            rel_crow = torch.arange(
                num_triplets + 1, device=dev, dtype=torch.long)
            partition["rel_matrix"] = torch.sparse_csr_tensor(
                crow_indices=rel_crow,
                col_indices=torch.from_numpy(sorted_relations).to(dev).long(),
                values=torch.ones(num_triplets, device=dev),
                size=(num_triplets, self.num_relations),
                device=dev
            )

            obj_crow = torch.arange(
                num_triplets + 1, device=dev, dtype=torch.long)
            partition["obj_matrix"] = torch.sparse_csr_tensor(
                crow_indices=obj_crow,
                col_indices=torch.from_numpy(sorted_tails).to(dev).long(),
                values=torch.ones(num_triplets, device=dev),
                size=(num_triplets, self.num_entities),
                device=dev
            )

        return partition

    # =========================================================================
    # CROSS-PARTITION HOP OPERATIONS
    # =========================================================================

    def _hop_across_partitions(
        self,
        frontier: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Perform one hop expansion across all relevant partitions."""
        if isinstance(frontier, torch.Tensor):
            active_indices = torch.nonzero(frontier).flatten().cpu().numpy()
        else:
            active_indices = np.where(frontier)[0]

        if len(active_indices) == 0:
            return frontier

        partition_groups = defaultdict(list)
        for entity_idx in active_indices:
            partition_id = self.partition_map[entity_idx]
            partition_groups[partition_id].append(entity_idx)

        if self.backend == self.BACKEND_TORCH:
            result = torch.zeros_like(frontier)
        else:
            result = np.zeros(self.num_entities, dtype=bool)

        for partition_id, entities in partition_groups.items():
            partition = self._get_partition(partition_id)
            partition_result = self._hop_in_partition(entities, partition)

            if self.backend == self.BACKEND_TORCH:
                result |= partition_result
            else:
                result |= partition_result

        return result

    def _hop_in_partition(
        self,
        entities: List[int],
        partition: Dict[str, Any]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Perform hop within a single partition."""
        if self.backend == self.BACKEND_SCIPY:
            vec = sp.csr_matrix(
                (np.ones(len(entities), bool),
                 (np.zeros(len(entities)), entities)),
                shape=(1, self.num_entities)
            )
            result_sparse = vec @ partition["sub_matrix"] @ partition["obj_matrix"]
            result = np.zeros(self.num_entities, dtype=bool)
            result[result_sparse.indices] = True
            return result

        elif self.backend == self.BACKEND_NUMBA:
            active = np.array(entities, dtype=np.int32)
            result = _numba_hop_kernel(
                active,
                partition["sub_indptr"],
                partition["sub_indices"],
                partition["obj_indices"],
                self.num_entities
            )
            return result.astype(bool)

        elif self.backend == self.BACKEND_TORCH:
            entities_t = torch.tensor(
                entities, dtype=torch.long, device=self.device)

            sub_crow = partition["sub_matrix"].crow_indices()
            sub_col = partition["sub_matrix"].col_indices()

            starts = sub_crow[entities_t]
            ends = sub_crow[entities_t + 1]
            counts = ends - starts

            if counts.sum() == 0:
                return torch.zeros(self.num_entities, dtype=torch.bool, device=self.device)

            row_ids = torch.repeat_interleave(torch.arange(
                len(starts), device=self.device), counts)
            cumsum = torch.cat(
                [torch.zeros(1, device=self.device, dtype=torch.long), counts[:-1].cumsum(0)])
            offsets = torch.arange(counts.sum().item(
            ), device=self.device) - torch.repeat_interleave(cumsum, counts)
            triplet_indices = sub_col[starts[row_ids] + offsets]

            obj_col = partition["obj_matrix"].col_indices()
            tail_entities = obj_col[triplet_indices]

            result = torch.zeros(
                self.num_entities, dtype=torch.bool, device=self.device)
            result[tail_entities] = True
            return result

    def _expand_paths_across_partitions(
        self,
        paths: Union[np.ndarray, torch.Tensor],
        visited: Optional[Union[np.ndarray, torch.Tensor]]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Expand paths across all relevant partitions."""
        if isinstance(paths, torch.Tensor):
            last_nodes = paths[:, -1].cpu().numpy()
        else:
            last_nodes = paths[:, -1]

        if len(last_nodes) == 0:
            return paths

        partition_groups = defaultdict(list)
        for path_idx, entity_idx in enumerate(last_nodes):
            partition_id = self.partition_map[entity_idx]
            partition_groups[partition_id].append(path_idx)

        all_expanded = []

        for partition_id, path_indices in partition_groups.items():
            partition = self._get_partition(partition_id)

            if self.backend == self.BACKEND_TORCH:
                partition_paths = paths[path_indices]
            else:
                partition_paths = paths[path_indices]

            expanded = self._expand_paths_in_partition(
                partition_paths, partition, visited)

            if len(expanded) > 0:
                all_expanded.append(expanded)

        if len(all_expanded) == 0:
            if self.backend == self.BACKEND_TORCH:
                return torch.tensor([], dtype=torch.long, device=self.device)
            else:
                return np.array([])

        if self.backend == self.BACKEND_TORCH:
            return torch.cat(all_expanded, dim=0)
        else:
            return np.vstack(all_expanded)

    def _expand_paths_in_partition(
        self,
        paths: Union[np.ndarray, torch.Tensor],
        partition: Dict[str, Any],
        visited: Optional[Union[np.ndarray, torch.Tensor]]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Expand paths within a single partition."""
        if self.backend == self.BACKEND_SCIPY:
            last_nodes = paths[:, -1]
            sub_slice = partition["sub_matrix"][last_nodes]
            triplet_indices = sub_slice.indices

            parents = np.repeat(np.arange(len(last_nodes)),
                                np.diff(sub_slice.indptr))

            rel_slice = partition["rel_matrix"][triplet_indices]
            relations = rel_slice.indices

            obj_slice = partition["obj_matrix"][triplet_indices]
            tails = obj_slice.indices

            valid = np.ones(len(tails), dtype=bool)
            if visited is not None:
                valid &= ~visited[tails]

            if not valid.any():
                return np.array([])

            return np.hstack([paths[parents[valid]], relations[valid].reshape(-1, 1), tails[valid].reshape(-1, 1)])

        elif self.backend == self.BACKEND_NUMBA:
            last_nodes = paths[:, -1]
            parents, triplet_indices = _numba_expand_kernel(
                last_nodes, partition["sub_indptr"], partition["sub_indices"])

            relations = partition["rel_indices"][triplet_indices]
            tails = partition["obj_indices"][triplet_indices]

            valid = np.ones(len(tails), dtype=bool)
            if visited is not None:
                valid &= ~visited[tails]

            if not valid.any():
                return np.array([])

            return np.hstack([paths[parents[valid]], relations[valid].reshape(-1, 1), tails[valid].reshape(-1, 1)])

        elif self.backend == self.BACKEND_TORCH:
            last_nodes = paths[:, -1]

            sub_crow = partition["sub_matrix"].crow_indices()
            sub_col = partition["sub_matrix"].col_indices()

            starts, ends = sub_crow[last_nodes], sub_crow[last_nodes + 1]
            counts = ends - starts
            total = counts.sum().item()

            if total == 0:
                return torch.tensor([], dtype=torch.long, device=self.device).reshape(0, paths.shape[1])

            parents = torch.repeat_interleave(torch.arange(
                len(last_nodes), device=self.device), counts)
            cumsum = torch.cat(
                [torch.zeros(1, device=self.device, dtype=torch.long), counts[:-1].cumsum(0)])
            offsets = torch.arange(total, device=self.device) - \
                torch.repeat_interleave(cumsum, counts)

            row_expanded = torch.repeat_interleave(
                torch.arange(len(starts), device=self.device), counts)
            triplet_indices = sub_col[starts[row_expanded] + offsets]

            rel_col = partition["rel_matrix"].col_indices()
            relations = rel_col[triplet_indices]

            obj_col = partition["obj_matrix"].col_indices()
            tails = obj_col[triplet_indices]

            valid = torch.ones(len(tails), dtype=torch.bool,
                               device=self.device)
            if visited is not None:
                valid &= ~visited[tails]

            if not valid.any():
                return torch.tensor([], dtype=torch.long, device=self.device)

            return torch.cat([paths[parents[valid]], relations[valid].unsqueeze(1), tails[valid].unsqueeze(1)], dim=1)

    # =========================================================================
    # BATCH PROCESSING WITH CACHE OPTIMIZATION
    # =========================================================================

    def batch_retrieve_at_k_hop(
        self,
        batch_entity_ids: List[List[str]],
        hops: int,
        shortest_path: bool = True
    ) -> List[List[str]]:
        """Batch retrieval with query reordering for cache efficiency."""
        return self._batch_process(
            batch_entity_ids,
            lambda ids: self.retrieve_at_k_hop(ids, hops, shortest_path)
        )

    def batch_retrieve_within_k_hop(
        self,
        batch_entity_ids: List[List[str]],
        hops: int,
        shortest_path: bool = True
    ) -> List[List[str]]:
        """Batch retrieval with query reordering for cache efficiency."""
        return self._batch_process(
            batch_entity_ids,
            lambda ids: self.retrieve_within_k_hop(ids, hops, shortest_path)
        )

    def batch_retrieve_with_paths_at_k_hop(
        self,
        batch_entity_ids: List[List[str]],
        hops: int = 2,
        shortest_path: bool = True,
        max_paths_per_entity: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Batch retrieval with query reordering for cache efficiency."""
        return self._batch_process(
            batch_entity_ids,
            lambda ids: self.retrieve_with_paths_at_k_hop(
                ids, hops, shortest_path, max_paths_per_entity)
        )

    def batch_retrieve_with_paths_within_k_hop(
        self,
        batch_entity_ids: List[List[str]],
        hops: int = 2,
        shortest_path: bool = True,
        max_paths_per_entity: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Batch retrieval with query reordering for cache efficiency."""
        return self._batch_process(
            batch_entity_ids,
            lambda ids: self.retrieve_with_paths_within_k_hop(
                ids, hops, shortest_path, max_paths_per_entity)
        )

    def _batch_process(self, batch_entity_ids: List[List[str]], process_fn) -> List:
        """Process batch with query reordering for cache efficiency."""
        if len(batch_entity_ids) == 0:
            return []

        query_info = self._analyze_query_partitions(batch_entity_ids)

        sorted_query_info = self._sort_by_partition_similarity(query_info)

        reordered_batch = [batch_entity_ids[q_idx]
                           for q_idx, _ in sorted_query_info]
        original_indices = [q_idx for q_idx, _ in sorted_query_info]

        reordered_results = [process_fn(entity_ids)
                             for entity_ids in reordered_batch]

        results = [None] * len(batch_entity_ids)
        for i, original_idx in enumerate(original_indices):
            results[original_idx] = reordered_results[i]

        return results

    def _analyze_query_partitions(self, batch_entities: List[List[str]]) -> List[Tuple[int, Set[int]]]:
        """Analyze which partitions each query needs."""
        query_info = []
        for i, entity_list in enumerate(batch_entities):
            required_partitions = set()
            for entity in entity_list:
                if entity in self.entity_to_idx:
                    entity_idx = self.entity_to_idx[entity]
                    if entity_idx in self.partition_map:
                        partition_id = self.partition_map[entity_idx]
                        required_partitions.add(partition_id)
            query_info.append((i, required_partitions))
        return query_info

    def _sort_by_partition_similarity(self, query_info: List[Tuple[int, Set[int]]]) -> List[Tuple[int, Set[int]]]:
        """Sort queries by partition requirements for better cache locality."""
        return sorted(query_info, key=lambda item: tuple(sorted(item[1])))

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _decode_paths(
        self,
        paths: Union[np.ndarray, torch.Tensor],
        max_paths: Optional[int]
    ) -> Dict[str, Any]:
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
                    decoded.extend(
                        [f"--{self.idx_to_relation[path[i]]}-->", self.idx_to_entity[path[i + 1]]])

            results[end].append(decoded)

        return {"entities": list(results.keys()), "paths": results}


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
    """One-hop expansion using CSR traversal."""
    result = np.zeros(num_entities, dtype=np.uint8)
    for i in range(len(active_indices)):
        entity = active_indices[i]
        for csr_pos in range(sub_indptr[entity], sub_indptr[entity + 1]):
            triplet_id = sub_indices[csr_pos]
            result[obj_indices[triplet_id]] = 1
    return result