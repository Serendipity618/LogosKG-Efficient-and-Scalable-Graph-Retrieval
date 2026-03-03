"""
Fast and Memory-Efficient Knowledge Graph Partitioner

Updated to work with NetworkX graphs stored as pickle files.
Supports both direct NetworkX graph input and triplet file input.
"""

import time
import pickle
import os
import networkx as nx
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


class KnowledgeGraphPartitioner:
    """
    Memory-efficient partitioner using two-pass algorithm.

    Now supports:
    1. NetworkX graph pickle files
    2. Original triplet text files

    Pass 1: Calculate degrees (keep only vertex->degree mapping)
    Pass 2: Stream through data and write to partitions

    Memory usage: O(V) where V is number of unique vertices
    """

    def __init__(self, input_path: str, output_dir: str, num_partitions: int,
                 input_type: str = "auto", encoding: str = "utf-8", batch_size: int = 10000):
        """
        Initialize the partitioner.

        Args:
            input_path: Path to input file (NetworkX pickle or triplet text file)
            output_dir: Directory to write partition files
            num_partitions: Number of partitions to create
            input_type: "networkx", "triplets", or "auto" (auto-detect)
            encoding: File encoding (for text files)
            batch_size: Number of triplets to buffer before writing
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.num_partitions = num_partitions
        self.input_type = input_type
        self.encoding = encoding
        self.batch_size = batch_size

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect input type if needed
        if self.input_type == "auto":
            self.input_type = self._detect_input_type()
            print(f"Auto-detected input type: {self.input_type}")

    def _detect_input_type(self) -> str:
        """Auto-detect if input is NetworkX pickle or triplet text file."""
        if self.input_path.suffix.lower() == '.pkl':
            return "networkx"
        elif self.input_path.suffix.lower() in ['.txt', '.tsv']:
            return "triplets"
        else:
            # Try to load as pickle first
            try:
                with open(self.input_path, 'rb') as f:
                    obj = pickle.load(f)
                    if hasattr(obj, 'edges') and hasattr(obj, 'nodes'):
                        return "networkx"
            except:
                pass
            return "triplets"

    # --- MINIMAL ADDITION: helper to coerce pickled content to a NetworkX graph ---
    def _to_networkx_graph(self, obj):
        """
        Coerce a pickled object into a NetworkX MultiDiGraph if it's not already a graph.
        Supports:
          - iterable of (u, v)
          - iterable of (u, v, data_dict)
          - iterable of (subject, relation, object) triplets
        """
        if hasattr(obj, "nodes") and hasattr(obj, "edges"):
            return obj  # already a NetworkX graph

        # Preserve duplicate triplets (parallel edges)
        G = nx.MultiDiGraph()
        try:
            for e in obj:
                if isinstance(e, (list, tuple)) and len(e) == 3 and isinstance(e[2], dict):
                    u, v, data = e
                    G.add_edge(str(u), str(v), **(data or {}))
                elif isinstance(e, (list, tuple)) and len(e) == 3:
                    s, r, o = e
                    G.add_edge(str(s), str(o), label=str(r))
                elif isinstance(e, (list, tuple)) and len(e) == 2:
                    u, v = e
                    G.add_edge(str(u), str(v))
        except TypeError:
            pass
        return G
    # ------------------------------------------------------------------------------

    def partition(self) -> Dict[str, int]:
        """
        Execute memory-efficient partitioning.

        Returns:
            Dictionary mapping each subject to its partition ID
        """
        total_start = time.time()

        print("=" * 60)
        print("MEMORY-EFFICIENT KNOWLEDGE GRAPH PARTITIONING")
        print(f"Input type: {self.input_type}")
        print("=" * 60)

        # Pass 1: Calculate degrees (memory: O(vertices))
        print("\nPass 1: Calculating vertex degrees...")
        step_start = time.time()
        vertex_degrees, total_triplets = self._calculate_degrees()
        print(f"  Time: {time.time() - step_start:.2f} seconds")
        print(f"  Found {len(vertex_degrees)} unique vertices")
        print(f"  Total triplets: {total_triplets}")

        # Assign partitions based on degrees (memory: O(vertices))
        print("\nAssigning vertices to partitions...")
        step_start = time.time()
        assignments = self._assign_partitions(vertex_degrees)
        print(f"  Time: {time.time() - step_start:.2f} seconds")

        # Free memory from degrees
        del vertex_degrees

        # Pass 2: Stream and write partitions
        print("\nPass 2: Writing partition files...")
        step_start = time.time()
        stats = self._stream_and_write_partitions(assignments)
        print(f"  Time: {time.time() - step_start:.2f} seconds")

        # Save metadata
        print("\nSaving metadata...")
        step_start = time.time()
        self._save_metadata(assignments, stats)
        print(f"  Time: {time.time() - step_start:.2f} seconds")

        # Calculate total time
        total_seconds = time.time() - total_start
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)

        print("\n" + "=" * 60)
        print(f"Partitioning complete!")
        print(f"Total time: {hours} hours {minutes} minutes {seconds} seconds")
        print("=" * 60)

        return assignments

    def _calculate_degrees(self) -> Tuple[Dict[str, int], int]:
        """
        Pass 1: Calculate degree for each vertex.
        Works with both NetworkX graphs and triplet files.
        """
        if self.input_type == "networkx":
            return self._calculate_degrees_networkx()
        else:
            return self._calculate_degrees_triplets()

    def _calculate_degrees_networkx(self) -> Tuple[Dict[str, int], int]:
        """Calculate degrees from NetworkX graph."""
        print("  Loading NetworkX graph...")

        # Load graph
        with open(self.input_path, 'rb') as f:
            kb = pickle.load(f)

        # Coerce to NetworkX graph if pickle holds edges/triplets
        kb = self._to_networkx_graph(kb)

        print(f"  Graph loaded: {kb.number_of_nodes()} nodes, {kb.number_of_edges()} edges")

        # Calculate degrees directly from graph
        vertex_degrees = {}

        # Get degree for each node (in-degree + out-degree for directed graphs)
        if kb.is_directed():
            for node in kb.nodes():
                in_deg = kb.in_degree(node)
                out_deg = kb.out_degree(node)
                vertex_degrees[str(node)] = in_deg + out_deg  # counts parallel edges in MultiDiGraph
        else:
            for node in kb.nodes():
                vertex_degrees[str(node)] = kb.degree(node)

        total_triplets = kb.number_of_edges()

        return vertex_degrees, total_triplets

    def _calculate_degrees_triplets(self) -> Tuple[Dict[str, int], int]:
        """Calculate degrees from triplet file (original implementation)."""
        out_degree = defaultdict(int)
        in_degree = defaultdict(int)
        total_triplets = 0

        # Stream through file
        with open(self.input_path, 'r', encoding=self.encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or '|' not in line:
                    continue

                parts = line.split('|')
                if len(parts) != 3:
                    continue

                subject, relation, obj = parts

                # Count degrees
                out_degree[subject] += 1
                in_degree[obj] += 1
                total_triplets += 1

                # Progress indicator
                if line_num % 100000000 == 0:
                    print(f"    Processed {line_num:,} lines...")

        # Combine degrees
        vertex_degrees = {}
        for vertex in set(out_degree.keys()) | set(in_degree.keys()):
            vertex_degrees[vertex] = out_degree.get(vertex, 0) + in_degree.get(vertex, 0)

        return vertex_degrees, total_triplets

    def _assign_partitions(self, vertex_degrees: Dict[str, int]) -> Dict[str, int]:
        """
        Assign vertices to partitions based on degrees.
        Uses balanced assignment without sorting all vertices.
        """
        assignments = {}
        partition_loads = [0] * self.num_partitions

        # Process high-degree vertices first (only top vertices)
        # This avoids sorting all vertices which would use too much memory
        threshold = sorted(vertex_degrees.values(), reverse=True)[min(10000, len(vertex_degrees) // 10)] if len(
            vertex_degrees) > 10000 else 0

        # Assign high-degree vertices with round-robin
        high_degree_vertices = [(v, d) for v, d in vertex_degrees.items() if d >= threshold]
        high_degree_vertices.sort(key=lambda x: x[1], reverse=True)

        for i, (vertex, degree) in enumerate(high_degree_vertices):
            # Assign to partition with minimum load
            min_partition = min(range(self.num_partitions), key=lambda p: partition_loads[p])
            assignments[vertex] = min_partition
            partition_loads[min_partition] += degree

        print(f"  Assigned {len(high_degree_vertices)} high-degree vertices")

        # Assign remaining vertices using simple hash
        for vertex in vertex_degrees:
            if vertex not in assignments:
                # Use hash for remaining vertices
                assignments[vertex] = hash(vertex) % self.num_partitions
                partition_loads[hash(vertex) % self.num_partitions] += vertex_degrees[vertex]

        # Print load distribution
        print(f"\n  Estimated load distribution:")
        for i, load in enumerate(partition_loads):
            print(f"    Partition {i}: {load:,} total degree")

        return assignments

    def _stream_and_write_partitions(self, assignments: Dict[str, int]) -> Dict:
        """
        Pass 2: Stream through data and write to partitions.
        Works with both NetworkX graphs and triplet files.
        """
        if self.input_type == "networkx":
            return self._stream_and_write_networkx(assignments)
        else:
            return self._stream_and_write_triplets(assignments)

    def _stream_and_write_networkx(self, assignments: Dict[str, int]) -> Dict:
        """Stream and write partitions from NetworkX graph."""
        print("  Loading NetworkX graph for partitioning...")

        # Load graph
        with open(self.input_path, 'rb') as f:
            kb = pickle.load(f)

        # Coerce to NetworkX graph if pickle holds edges/triplets
        kb = self._to_networkx_graph(kb)

        # Initialize partition writers with buffers
        partition_buffers = [[] for _ in range(self.num_partitions)]
        partition_counts = [0] * self.num_partitions

        # Open all partition files
        partition_files = []
        for i in range(self.num_partitions):
            f = open(self.output_dir / f"part_{i}.pkl.tmp", 'wb')
            partition_files.append(f)

        try:
            # Extract triplets and write to partitions
            processed = 0
            # Iterate multiedges to preserve duplicates
            # Handle both regular graphs and multigraphs
            if isinstance(kb, (nx.MultiGraph, nx.MultiDiGraph)):
                edge_iter = kb.edges(keys=True, data=True)
            else:
                edge_iter = ((u, v, None, data) for u, v, data in kb.edges(data=True))

            for u, v, key, data in edge_iter:
                # Convert to triplet format
                subject = str(u)
                relation = (data or {}).get('label', 'related_to')
                obj = str(v)

                # Get partition for this triplet
                partition_id = assignments.get(subject, hash(subject) % self.num_partitions)

                # Add to buffer
                partition_buffers[partition_id].append((subject, relation, obj))
                partition_counts[partition_id] += 1

                # Write buffer if full
                if len(partition_buffers[partition_id]) >= self.batch_size:
                    pickle.dump(partition_buffers[partition_id], partition_files[partition_id])
                    partition_buffers[partition_id] = []

                processed += 1

                # Progress indicator
                if processed % 1_000_000 == 0:
                    print(f"    Processed {processed:,} edges...")

            # Write remaining buffers
            for i in range(self.num_partitions):
                if partition_buffers[i]:
                    pickle.dump(partition_buffers[i], partition_files[i])

        finally:
            for f in partition_files:
                f.close()

        # Consolidate temporary files into final pickle files
        print("\n  Consolidating partition files...")
        for i in range(self.num_partitions):
            self._consolidate_partition(i)
            print(f"    Partition {i}: {partition_counts[i]:,} triplets")

        # Calculate statistics
        stats = {
            'partition_sizes': {i: partition_counts[i] for i in range(self.num_partitions)},
            'total_triplets': sum(partition_counts),
            'num_partitions': self.num_partitions,
            'avg_triplets_per_partition': sum(partition_counts) / self.num_partitions if self.num_partitions else 0,
            'max_triplets_per_partition': max(partition_counts) if partition_counts else 0,
            'min_triplets_per_partition': min(partition_counts) if partition_counts else 0
        }

        # Check balance
        if partition_counts and stats['avg_triplets_per_partition'] > 0:
            imbalance = (max(partition_counts) - min(partition_counts)) / stats['avg_triplets_per_partition'] * 100
            print(f"\n  Partition balance: {imbalance:.1f}% difference between max and min")

        return stats

    def _stream_and_write_triplets(self, assignments: Dict[str, int]) -> Dict:
        """Stream and write partitions from triplet file (original implementation)."""
        # Initialize partition writers with buffers
        partition_buffers = [[] for _ in range(self.num_partitions)]
        partition_counts = [0] * self.num_partitions

        # Open all partition files
        partition_files = []
        for i in range(self.num_partitions):
            f = open(self.output_dir / f"part_{i}.pkl.tmp", 'wb')
            partition_files.append(f)

        try:
            # Stream through input file
            with open(self.input_path, 'r', encoding=self.encoding) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or '|' not in line:
                        continue

                    parts = line.split('|')
                    if len(parts) != 3:
                        continue

                    subject, relation, obj = parts

                    # Get partition for this triplet
                    partition_id = assignments.get(subject, hash(subject) % self.num_partitions)

                    # Add to buffer
                    partition_buffers[partition_id].append((subject, relation, obj))
                    partition_counts[partition_id] += 1

                    # Write buffer if full
                    if len(partition_buffers[partition_id]) >= self.batch_size:
                        pickle.dump(partition_buffers[partition_id], partition_files[partition_id])
                        partition_buffers[partition_id] = []

                    # Progress indicator
                    if line_num % 100000000 == 0:
                        print(f"    Processed {line_num:,} lines...")

            # Write remaining buffers
            for i in range(self.num_partitions):
                if partition_buffers[i]:
                    pickle.dump(partition_buffers[i], partition_files[i])

        finally:
            for f in partition_files:
                f.close()

        # Consolidate temporary files into final pickle files
        print("\n  Consolidating partition files...")
        for i in range(self.num_partitions):
            self._consolidate_partition(i)
            print(f"    Partition {i}: {partition_counts[i]:,} triplets")

        # Calculate statistics
        stats = {
            'partition_sizes': {i: partition_counts[i] for i in range(self.num_partitions)},
            'total_triplets': sum(partition_counts),
            'num_partitions': self.num_partitions,
            'avg_triplets_per_partition': sum(partition_counts) / self.num_partitions,
            'max_triplets_per_partition': max(partition_counts),
            'min_triplets_per_partition': min(partition_counts)
        }

        # Check balance
        if partition_counts:
            imbalance = (max(partition_counts) - min(partition_counts)) / stats['avg_triplets_per_partition'] * 100
            print(f"\n  Partition balance: {imbalance:.1f}% difference between max and min")

        return stats

    def _consolidate_partition(self, partition_id: int):
        """
        Consolidate temporary partition file into final pickle file.
        Reads batches and combines them into single pickle.
        """
        temp_file = self.output_dir / f"part_{partition_id}.pkl.tmp"
        final_file = self.output_dir / f"part_{partition_id}.pkl"

        all_triplets = []

        # Read all batches from temp file
        if temp_file.exists():
            with open(temp_file, 'rb') as f:
                while True:
                    try:
                        batch = pickle.load(f)
                        all_triplets.extend(batch)
                    except EOFError:
                        break

        # Write as a single pickle
        with open(final_file, 'wb') as f:
            pickle.dump(all_triplets, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Remove temp file
        if temp_file.exists():
            os.remove(temp_file)

    def _save_metadata(self, assignments: Dict[str, int], stats: Dict):
        """Save metadata as pickle files."""
        # Save subject-to-partition mapping
        assignments_file = self.output_dir / "subject_to_partition.pkl"
        with open(assignments_file, 'wb') as f:
            pickle.dump(assignments, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved {len(assignments):,} vertex assignments")

        # Save statistics
        stats_file = self.output_dir / "partition_stats.pkl"
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved partition statistics")
