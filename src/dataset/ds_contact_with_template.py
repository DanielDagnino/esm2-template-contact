from typing import List
import os
import glob
import numpy as np
from path import Path

from torch.utils.data import Dataset

from dataset.mmseqs_parse import load_mmseqs_tsv, topk_hits_for


class ContactWithTemplateDataset(Dataset):
    def __init__(
            self,
            processed_dir: str,
            split: str,
            mmseqs_tsv: str,
            top_k: int,
            bin_edges: List[float] = (6., 8., 10., 14.),
    ) -> None:
        """Dataset for protein contact maps with template priors from MMseqs2 hits.
        Args:
            processed_dir: Directory containing processed .npz files.
            split: Dataset split name ("train", "test").
            mmseqs_tsv: Path to MMseqs2 TSV file with search results.
            top_k: Number of top templates to use for priors.
            bin_edges: Distance bin edges for discretizing distances.
        """

        self.dir = Path(processed_dir).expanduser() / split
        self.files = sorted(glob.glob(os.path.join(self.dir, "*.npz")))
        self.top_k = top_k
        self.bin_edges = bin_edges

        mmseqs_tsv = Path(mmseqs_tsv).expanduser()
        if os.path.exists(mmseqs_tsv):
            self.mmseqs = load_mmseqs_tsv(mmseqs_tsv)
        else:
            raise FileNotFoundError(f"MMseqs2 TSV file not found: {mmseqs_tsv}")

        if len(self.files) == 0:
            raise ValueError(f"No data files found in {self.dir}")

    def __len__(self) -> int:
        return len(self.files)

    def passes_filters(
            self, 
            hit: dict, 
            min_id: float,
            min_cov: float,
    ) -> bool:
        """Check if a MMseqs2 hit passes identity and coverage filters.
        Args:
             hit: MMseqs2 hit dictionary.
             min_id: Minimum sequence identity threshold.
             min_cov: Minimum coverage threshold.
        Returns:
            True if the hit passes the filters, False otherwise.
        """
        pident = float(hit["pident"]) / 100.0  # sequence identity
        q_cov = (int(hit["qend"]) - int(hit["qstart"]) + 1) / float(hit["qlen"])  # query coverage
        t_cov = (int(hit["tend"]) - int(hit["tstart"]) + 1) / float(hit["tlen"])  # template coverage
        return pident >= min_id and q_cov >= min_cov and t_cov >= min_cov
        
    def _load_templates(
            self,
            qname: str,
            seq_len: int,
            min_identity: float = 0.2,
            min_coverage: float = 0.5,
            max_candidates: int = 200,
            relax_factor: float = 0.8,
    ) -> (np.ndarray, np.ndarray):
        """Build template priors for one query sequence.
        Args:
            qname: Query sequence name.
            seq_len: Length of the query sequence.
            min_identity: Minimum sequence identity threshold. Recommended 0.2 to ensure structural similarity.
            min_coverage: Minimum coverage threshold. Recommended 0.5 to ensures the alignment covers most of the region
            max_candidates: Maximum number of MMseqs hits to consider. Recommended 200.
            relax_factor: Factor to relax thresholds if no hits pass. Recommended 0.8.
        Returns:
            pri_contact: Prior contact map (seq_len x seq_len).
            pri_bins: Prior distance bins (seq_len x seq_len x num_bins).
        """

        """Build template priors for one query sequence."""
        pri_contact = np.zeros((seq_len, seq_len), dtype=np.float32)
        pri_bins = np.zeros((seq_len, seq_len, len(self.bin_edges) + 1), dtype=np.float32)

        # Over-fetch many MMseqs hits, already sorted by bitscore/pident
        hits_all = topk_hits_for(qname, self.mmseqs, topk=max_candidates)

        n_relax_max = 2
        accepted = 0
        weight_sum = 0.0
        for relax_idx in range(n_relax_max):
            for _, hit in hits_all.iterrows():
                pident = float(hit["pident"]) / 100.

                # Skip until filters pass (with possible relaxed thresholds)
                if not self.passes_filters(hit, min_identity, min_coverage):
                    continue

                # Load template data
                tname = hit["target"]
                tpath = os.path.join(os.path.dirname(self.dir), "train", tname)
                if not os.path.exists(tpath):
                    raise FileNotFoundError(f"Template file not found: {tpath}")

                tnpz = np.load(tpath, allow_pickle=True)
                qs, qe = int(hit["qstart"]) - 1, int(hit["qend"]) - 1  # query indices
                ts, te = int(hit["tstart"]) - 1, int(hit["tend"]) - 1  # template indices

                # skip if query/template indices out of bounds
                if qs < 0 or ts < 0 or qe >= seq_len:
                    continue

                tL = int(tnpz["contact"].shape[0])  # template length
                te = min(te, tL - 1)  # adjust template end if needed

                # skip if lengths disagree
                if te - ts != qe - qs:
                    continue

                # extract template contact and distance
                t_contact = tnpz["contact"][ts:te + 1, ts:te + 1].astype(np.float32)
                t_dist = tnpz["dist"][ts:te + 1, ts:te + 1].astype(np.float32)

                # Discretize distances into bins
                bins = np.digitize(t_dist, bins=np.array(self.bin_edges, dtype=np.float32))
                onehot = np.eye(len(self.bin_edges) + 1, dtype=np.float32)[bins]

                # Soft weighting: exponential emphasis on higher identity
                w = np.exp(pident / 0.1)
                pri_contact[qs:qe + 1, qs:qe + 1] += w * t_contact
                pri_bins[qs:qe + 1, qs:qe + 1, :] += w * onehot
                weight_sum += w
                accepted += 1

                # Stop if we have enough templates
                if accepted >= self.top_k:
                    break

            # Stop if enough templates accepted or max relax reached
            if (accepted >= self.top_k) or (relax_idx == n_relax_max - 1):
                break
            else:
                # Relax filters and retry
                min_identity *= relax_factor
                min_coverage *= relax_factor

        # Normalize by total weight
        if weight_sum > 0:
            pri_contact /= weight_sum
            pri_bins /= weight_sum

        return pri_contact, pri_bins

    def __getitem__(self, idx: int) -> dict:
        # load data
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)

        # extract fields
        seq = str(data["seq"])
        contact = data["contact"].astype(np.float32)
        dist = data["dist"].astype(np.float32)
        seq_len = len(seq)

        # load template priors
        qname = os.path.basename(path)
        pri_contact, pri_bins = self._load_templates(qname, seq_len)

        # pack items
        item = {
            "name": os.path.basename(path),
            "seq": seq,
            "seq_len": seq_len,
            "contact": contact,
            "dist": dist,
            "pri_contact": pri_contact,
            "pri_bins": pri_bins,
        }
        return item
