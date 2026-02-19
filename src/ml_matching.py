import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _term_lists_from_pairs(
    pairs: List[Dict],
    bank_df: Any,
    register_df: Any,
) -> Tuple[List[str], List[str], List[List[str]], List[List[str]]]:
    bank_terms_per_doc = []
    reg_terms_per_doc = []
    all_bank_terms = set()
    all_reg_terms = set()
    for p in pairs:
        bid = p.get("bank_id") or p.get("bank_tx", {}).get("transaction_id")
        rid = p.get("register_id") or p.get("register_tx", {}).get("transaction_id")
        if bid is None or rid is None:
            continue
        brow = bank_df[bank_df["transaction_id"] == bid]
        rrow = register_df[register_df["transaction_id"] == rid]
        if brow.empty or rrow.empty:
            continue
        bterms = brow.iloc[0].get("terms", [])
        rterms = rrow.iloc[0].get("terms", [])
        if not isinstance(bterms, list):
            bterms = list(bterms) if hasattr(bterms, "__iter__") else []
        if not isinstance(rterms, list):
            rterms = list(rterms) if hasattr(rterms, "__iter__") else []
        bank_terms_per_doc.append(bterms)
        reg_terms_per_doc.append(rterms)
        all_bank_terms.update(bterms)
        all_reg_terms.update(rterms)
    bank_vocab = sorted(all_bank_terms)
    reg_vocab = sorted(all_reg_terms)
    return bank_vocab, reg_vocab, bank_terms_per_doc, reg_terms_per_doc


def _cooccurrence_and_marginals(
    bank_terms_per_doc: List[List[str]],
    reg_terms_per_doc: List[List[str]],
    bank_vocab: List[str],
    reg_vocab: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    B, R = len(bank_vocab), len(reg_vocab)
    bank2i = {t: i for i, t in enumerate(bank_vocab)}
    reg2i = {t: i for i, t in enumerate(reg_vocab)}
    cooc = np.zeros((B, R))
    for bterms, rterms in zip(bank_terms_per_doc, reg_terms_per_doc):
        for b in bterms:
            if b not in bank2i:
                continue
            for r in rterms:
                if r not in reg2i:
                    continue
                cooc[bank2i[b], reg2i[r]] += 1
    bank_count = cooc.sum(axis=1)
    reg_count = cooc.sum(axis=0)
    return cooc, bank_count, reg_count


def _mutual_information(
    cooc: np.ndarray,
    bank_count: np.ndarray,
    reg_count: np.ndarray,
) -> np.ndarray:
    total = cooc.sum()
    if total <= 0:
        return np.zeros_like(cooc)
    p_br = cooc / total
    p_b = bank_count / total
    p_r = reg_count / total
    mi = np.zeros_like(cooc)
    for i in range(cooc.shape[0]):
        for j in range(cooc.shape[1]):
            if p_br[i, j] <= 0:
                continue
            if p_b[i] <= 0 or p_r[j] <= 0:
                continue
            mi[i, j] = p_br[i, j] * math.log(
                p_br[i, j] / (p_b[i] * p_r[j] + 1e-12) + 1e-12
            )
    return mi


def _term_document_matrix(
    reg_terms_per_doc: List[List[str]],
    reg_vocab: List[str],
) -> np.ndarray:
    R, D = len(reg_vocab), len(reg_terms_per_doc)
    reg2i = {t: i for i, t in enumerate(reg_vocab)}
    M = np.zeros((R, D))
    for d, rterms in enumerate(reg_terms_per_doc):
        for t in rterms:
            if t in reg2i:
                M[reg2i[t], d] += 1
    return M


def _project_bank_to_latent(
    term_vec: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    """term_vec (B,) @ U (B x k) -> (k,)"""
    return term_vec.dot(U)


def _project_register_to_latent(
    term_vec: np.ndarray,
    W: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    in_bank_space = W.dot(term_vec)
    return in_bank_space.dot(U)


def fit_svd_model(
    validated_pairs: List[Dict],
    bank_df: Any,
    register_df: Any,
    n_components: int = 30,
) -> Dict[str, Any]:
    if not validated_pairs:
        return {
            "bank_vocab": [],
            "reg_vocab": [],
            "W": np.array([[]]),
            "U": np.array([[]]),
            "n_components": n_components,
        }
    bank_vocab, reg_vocab, bank_terms_per_doc, reg_terms_per_doc = _term_lists_from_pairs(
        validated_pairs, bank_df, register_df
    )
    if not bank_vocab or not reg_vocab:
        return {
            "bank_vocab": bank_vocab or [],
            "reg_vocab": reg_vocab or [],
            "W": np.array([[]]),
            "U": np.array([[]]),
            "n_components": n_components,
        }
    cooc, bank_count, reg_count = _cooccurrence_and_marginals(
        bank_terms_per_doc, reg_terms_per_doc, bank_vocab, reg_vocab
    )
    W = _mutual_information(cooc, bank_count, reg_count)
    M_reg = _term_document_matrix(reg_terms_per_doc, reg_vocab)
    X = W.dot(M_reg)
    k = min(n_components, min(X.shape) - 1, 50)
    if k < 1:
        U = np.eye(len(bank_vocab), 1)
        return {
            "bank_vocab": bank_vocab,
            "reg_vocab": reg_vocab,
            "W": W,
            "U": U,
            "n_components": 1,
        }
    try:
        U_full, s, _ = np.linalg.svd(X, full_matrices=False)
        U = U_full[:, :k]
    except Exception:
        U = np.eye(len(bank_vocab), min(k, len(bank_vocab)))
    return {
        "bank_vocab": bank_vocab,
        "reg_vocab": reg_vocab,
        "W": W,
        "U": U,
        "n_components": U.shape[1],
    }


def _tx_to_term_vector(terms: List, vocab: List[str]) -> np.ndarray:
    v = np.zeros(len(vocab))
    t2i = {t: i for i, t in enumerate(vocab)}
    for t in terms:
        if t in t2i:
            v[t2i[t]] += 1
    return v


def match_remaining_with_ml(
    bank_df: Any,
    register_df: Any,
    validated_pairs: List[Dict],
    excluded_bank_ids: set,
    excluded_register_ids: set,
    n_components: int = 30,
    confidence_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    model = fit_svd_model(validated_pairs, bank_df, register_df, n_components=n_components)
    bank_vocab = model["bank_vocab"]
    reg_vocab = model["reg_vocab"]
    W = model["W"]
    U = model["U"]
    if not bank_vocab or not reg_vocab or U.size == 0:
        return []

    bank_df = bank_df[~bank_df["transaction_id"].isin(excluded_bank_ids)].copy()
    register_df = register_df[
        ~register_df["transaction_id"].isin(excluded_register_ids)
    ].copy()
    if bank_df.empty or register_df.empty:
        return []

    # Project all bank and register transactions to latent space
    bank_vectors = []
    bank_ids = []
    for _, row in bank_df.iterrows():
        bvec = _tx_to_term_vector(row.get("terms", []), bank_vocab)
        proj = _project_bank_to_latent(bvec, U)
        bank_vectors.append(proj)
        bank_ids.append(row["transaction_id"])
    bank_vectors = np.array(bank_vectors)
    if bank_vectors.size == 0:
        return []

    reg_vectors = []
    reg_ids = []
    for _, row in register_df.iterrows():
        rvec = _tx_to_term_vector(row.get("terms", []), reg_vocab)
        proj = _project_register_to_latent(rvec, W, U)
        reg_vectors.append(proj)
        reg_ids.append(row["transaction_id"])
    reg_vectors = np.array(reg_vectors)
    if reg_vectors.size == 0:
        return []

    # Cosine similarity: (n_bank x dim) @ (dim x n_reg) -> (n_bank x n_reg)
    sim = cosine_similarity(bank_vectors, reg_vectors)
    used_reg = set()
    results = []
    for i in range(bank_vectors.shape[0]):
        best_j = None
        best_sim = confidence_threshold
        for j in range(reg_vectors.shape[0]):
            if reg_ids[j] in used_reg:
                continue
            if sim[i, j] > best_sim:
                best_sim = sim[i, j]
                best_j = j
        if best_j is not None:
            used_reg.add(reg_ids[best_j])
            bank_row = bank_df[bank_df["transaction_id"] == bank_ids[i]].iloc[0]
            reg_row = register_df[
                register_df["transaction_id"] == reg_ids[best_j]
            ].iloc[0]
            results.append({
                "bank_id": bank_ids[i],
                "register_id": reg_ids[best_j],
                "bank_tx": bank_row.drop(labels=["terms"], errors="ignore").to_dict(),
                "register_tx": reg_row.drop(labels=["terms"], errors="ignore").to_dict(),
                "confidence": float(best_sim),
                "flags": [],
                "source": "ml_svd",
            })
    return results
