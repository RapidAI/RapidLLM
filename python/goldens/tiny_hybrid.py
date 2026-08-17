#!/usr/bin/env python3
"""Tiny hybrid fixture + golden tokens.

Implements the same HF-locked math as RapidLLM C++ (no torch required).
Writes a mini HF dir, a mini GGUF, invalid loaders, and golden.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
from pathlib import Path


H = 32
V = 48
L = 8
INTER = 64
NK, NV, DK, DV = 2, 6, 8, 8
NQ, NKV, HD = 4, 2, 8
CONV_K = 4
EPS = 1e-6
THETA = 10000.0
ROT = 2
SEED = 7
PROMPT = [1, 2, 3]
N_NEW = 6


def rng():
    r = random.Random(SEED)
    return r


def randn(r, n, scale=0.05):
    return [r.gauss(0.0, scale) for _ in range(n)]


def mat(r, rows, cols, scale=0.05):
    return [randn(r, cols, scale) for _ in range(rows)]


def vec(r, n, scale=0.05):
    return randn(r, n, scale)


def silu(x):
    return x / (1.0 + math.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def softplus(x):
    ax = abs(x)
    return math.log1p(math.exp(-ax)) + (x if x > 0 else 0.0)


def rmsnorm(x, w, eps=EPS, plus_one=True):
    acc = sum(t * t for t in x) / len(x)
    inv = 1.0 / math.sqrt(acc + eps)
    if plus_one:
        return [xi * inv * (1.0 + w[i]) for i, xi in enumerate(x)]
    return [xi * inv * w[i] for i, xi in enumerate(x)]


def gated_rmsnorm(x, z, g, eps=EPS):
    acc = sum(t * t for t in x) / len(x)
    inv = 1.0 / math.sqrt(acc + eps)
    return [g[i] * (x[i] * inv) * silu(z[i]) for i in range(len(x))]


def l2norm(x, eps=EPS):
    acc = math.sqrt(sum(t * t for t in x) + eps)
    return [t / acc for t in x]


def gemv(W, x):
    return [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]


def conv1d_prefill(seq_x, w, dim, k=CONV_K):
    # seq_x: list of dim vectors; w: [dim][k]
    out = []
    for t in range(len(seq_x)):
        yt = []
        for c in range(dim):
            acc = 0.0
            for p in range(k):
                src = t - (k - 1 - p)
                xv = seq_x[src][c] if src >= 0 else 0.0
                acc += w[c][p] * xv
            yt.append(silu(acc))
        out.append(yt)
    return out


def rope(q_heads, k_heads, pos):
    def apply(heads):
        out = []
        for h in heads:
            v = h[:]
            for i in range(ROT // 2):
                freq = 1.0 / (THETA ** (i / (ROT / 2)))
                ang = pos * freq
                c, s = math.cos(ang), math.sin(ang)
                x0, x1 = v[2 * i], v[2 * i + 1]
                v[2 * i] = x0 * c - x1 * s
                v[2 * i + 1] = x0 * s + x1 * c
            out.append(v)
        return out

    return apply(q_heads), apply(k_heads)


def delta_step(S, q, k, v, beta, g_log):
    # S[h][dk][dv]
    o = []
    for h in range(NV):
        qn = l2norm(q[h])
        kn = l2norm(k[h])
        scale = 1.0 / math.sqrt(DK)
        qn = [t * scale for t in qn]
        g = math.exp(g_log[h])
        for i in range(DK):
            for j in range(DV):
                S[h][i][j] *= g
        kv = [0.0] * DV
        for i in range(DK):
            ki = kn[i]
            for j in range(DV):
                kv[j] += S[h][i][j] * ki
        delta = [(v[h][j] - kv[j]) * beta[h] for j in range(DV)]
        for i in range(DK):
            for j in range(DV):
                S[h][i][j] += kn[i] * delta[j]
        oh = [0.0] * DV
        for i in range(DK):
            for j in range(DV):
                oh[j] += S[h][i][j] * qn[i]
        o.append(oh)
    return o


def attn_scores(qs, ks, vs):
    # qs: [t][nq][hd], causal
    seq = len(qs)
    out = [[[0.0] * HD for _ in range(NQ)] for _ in range(seq)]
    scale = 1.0 / math.sqrt(HD)
    rep = NQ // NKV
    for t in range(seq):
        for hq in range(NQ):
            hkv = hq // rep
            scores = []
            m = -1e30
            for s in range(t + 1):
                acc = sum(qs[t][hq][d] * ks[s][hkv][d] for d in range(HD)) * scale
                scores.append(acc)
                m = max(m, acc)
            z = sum(math.exp(s - m) for s in scores)
            for s, sc in enumerate(scores):
                a = math.exp(sc - m) / z
                for d in range(HD):
                    out[t][hq][d] += a * vs[s][hkv][d]
    return out


def make_weights(r):
    W = {}
    W["embed"] = mat(r, V, H, 0.08)
    W["lm_head"] = mat(r, V, H, 0.08)
    W["final_norm"] = vec(r, H, 0.02)
    kinds = ["D", "D", "D", "A"] * 2
    for i, k in enumerate(kinds):
        p = f"L{i}"
        W[p + ".attn_norm"] = vec(r, H, 0.02)
        W[p + ".ffn_norm"] = vec(r, H, 0.02)
        W[p + ".gate"] = mat(r, INTER, H, 0.05)
        W[p + ".up"] = mat(r, INTER, H, 0.05)
        W[p + ".down"] = mat(r, H, INTER, 0.05)
        if k == "D":
            qkv = NK * DK + NK * DK + NV * DV
            W[p + ".qkv"] = mat(r, qkv, H, 0.04)
            W[p + ".z"] = mat(r, NV * DV, H, 0.04)
            W[p + ".out"] = mat(r, H, NV * DV, 0.04)
            W[p + ".a"] = mat(r, NV, H, 0.04)
            W[p + ".b"] = mat(r, NV, H, 0.04)
            W[p + ".A_log"] = [-0.5 + 0.02 * r.random() for _ in range(NV)]
            W[p + ".dt"] = [0.1 * r.random() for _ in range(NV)]
            W[p + ".conv"] = [vec(r, CONV_K, 0.05) for _ in range(qkv)]
            W[p + ".gn"] = [1.0 + 0.02 * r.gauss(0, 1) for _ in range(NV * DV)]
        else:
            W[p + ".wq"] = mat(r, NQ * HD * 2, H, 0.04)
            W[p + ".wk"] = mat(r, NKV * HD, H, 0.04)
            W[p + ".wv"] = mat(r, NKV * HD, H, 0.04)
            W[p + ".wo"] = mat(r, H, NQ * HD, 0.04)
            W[p + ".qn"] = vec(r, HD, 0.02)
            W[p + ".kn"] = vec(r, HD, 0.02)
    W["mtp.fc"] = mat(r, H, 2 * H, 0.04)
    W["mtp.norm"] = vec(r, H, 0.02)
    W["mtp.pre_h"] = vec(r, H, 0.02)
    W["mtp.pre_e"] = vec(r, H, 0.02)
    W["mtp.attn_norm"] = vec(r, H, 0.02)
    W["mtp.ffn_norm"] = vec(r, H, 0.02)
    W["mtp.wq"] = mat(r, NQ * HD * 2, H, 0.04)
    W["mtp.wk"] = mat(r, NKV * HD, H, 0.04)
    W["mtp.wv"] = mat(r, NKV * HD, H, 0.04)
    W["mtp.wo"] = mat(r, H, NQ * HD, 0.04)
    W["mtp.qn"] = vec(r, HD, 0.02)
    W["mtp.kn"] = vec(r, HD, 0.02)
    W["mtp.gate"] = mat(r, INTER, H, 0.05)
    W["mtp.up"] = mat(r, INTER, H, 0.05)
    W["mtp.down"] = mat(r, H, INTER, 0.05)
    return W, kinds


def forward(W, kinds, ids, n_new):
    # caches
    S = {}
    conv_state = {}
    k_cache = {}
    v_cache = {}
    for i, k in enumerate(kinds):
        if k == "D":
            S[i] = [[[0.0] * DV for _ in range(DK)] for _ in range(NV)]
            conv_state[i] = [[0.0] * CONV_K for _ in range(NK * DK + NK * DK + NV * DV)]
        else:
            k_cache[i] = []
            v_cache[i] = []

    def embed(tok):
        return W["embed"][tok][:]

    def one_layer_delta(i, x, pos, prefill_seq=None):
        p = f"L{i}"
        xn = rmsnorm(x, W[p + ".attn_norm"])
        qkv = gemv(W[p + ".qkv"], xn)
        z = gemv(W[p + ".z"], xn)
        a = gemv(W[p + ".a"], xn)
        b = gemv(W[p + ".b"], xn)
        dim = len(qkv)
        # conv update
        for c in range(dim):
            row = conv_state[i][c]
            for t in range(CONV_K - 1):
                row[t] = row[t + 1]
            row[CONV_K - 1] = qkv[c]
        mixed = []
        for c in range(dim):
            acc = sum(W[p + ".conv"][c][t] * conv_state[i][c][t] for t in range(CONV_K))
            mixed.append(silu(acc))
        qdim = NK * DK
        Q = mixed[:qdim]
        K = mixed[qdim : 2 * qdim]
        V = mixed[2 * qdim :]
        qh, kh, vh = [], [], []
        beta, glog = [], []
        for h in range(NV):
            src = h // (NV // NK)
            qh.append(Q[src * DK : (src + 1) * DK])
            kh.append(K[src * DK : (src + 1) * DK])
            vh.append(V[h * DV : (h + 1) * DV])
            beta.append(sigmoid(b[h]))
            glog.append(-math.exp(W[p + ".A_log"][h]) * softplus(a[h] + W[p + ".dt"][h]))
        o = delta_step(S[i], qh, kh, vh, beta, glog)
        oflat = [o[h][d] for h in range(NV) for d in range(DV)]
        og = gated_rmsnorm(oflat, z, W[p + ".gn"])
        y = gemv(W[p + ".out"], og)
        return [x[d] + y[d] for d in range(H)]

    def one_layer_attn(i, x, pos):
        p = f"L{i}"
        xn = rmsnorm(x, W[p + ".attn_norm"])
        qg = gemv(W[p + ".wq"], xn)
        k = gemv(W[p + ".wk"], xn)
        v = gemv(W[p + ".wv"], xn)
        q = qg[: NQ * HD]
        gate = qg[NQ * HD :]
        qh = [rmsnorm(q[h * HD : (h + 1) * HD], W[p + ".qn"]) for h in range(NQ)]
        kh = [rmsnorm(k[h * HD : (h + 1) * HD], W[p + ".kn"]) for h in range(NKV)]
        vh = [v[h * HD : (h + 1) * HD] for h in range(NKV)]
        qh, kh = rope(qh, kh, pos)
        k_cache[i].append(kh)
        v_cache[i].append(vh)
        # decode-style over cache (also used during prefill token-by-token)
        qs = [qh]
        # build full seq q from just current — use cache k/v
        scale = 1.0 / math.sqrt(HD)
        rep = NQ // NKV
        o = []
        for hq in range(NQ):
            hkv = hq // rep
            scores = []
            m = -1e30
            for s in range(pos + 1):
                acc = sum(qh[hq][d] * k_cache[i][s][hkv][d] for d in range(HD)) * scale
                scores.append(acc)
                m = max(m, acc)
            z = sum(math.exp(s - m) for s in scores)
            oh = [0.0] * HD
            for s, sc in enumerate(scores):
                a = math.exp(sc - m) / z
                for d in range(HD):
                    oh[d] += a * v_cache[i][s][hkv][d]
            g = [sigmoid(gate[hq * HD + d]) for d in range(HD)]
            o.extend([oh[d] * g[d] for d in range(HD)])
        y = gemv(W[p + ".wo"], o)
        return [x[d] + y[d] for d in range(H)]

    def mlp(i, x):
        p = f"L{i}"
        xn = rmsnorm(x, W[p + ".ffn_norm"])
        g = gemv(W[p + ".gate"], xn)
        u = gemv(W[p + ".up"], xn)
        h = [silu(g[j]) * u[j] for j in range(INTER)]
        y = gemv(W[p + ".down"], h)
        return [x[d] + y[d] for d in range(H)]

    last_h = [0.0] * H
    all_h = []

    def step(tok, pos):
        nonlocal last_h
        x = embed(tok)
        for i, knd in enumerate(kinds):
            if knd == "D":
                x = one_layer_delta(i, x, pos)
            else:
                x = one_layer_attn(i, x, pos)
            x = mlp(i, x)
        all_h.append(x[:])  # last-layer residual for MTP hist
        xn = rmsnorm(x, W["final_norm"])
        last_h = xn[:]
        logits = gemv(W["lm_head"], xn)
        return logits

    # token-by-token prefill so conv/attn caches stay aligned with C++ decode path
    logits = None
    for t, tok in enumerate(ids):
        logits = step(tok, t)
    hidden_after_prompt = last_h[:]
    prompt_h = [row[:] for row in all_h]
    pos = len(ids)
    out = []
    for _ in range(n_new):
        best = max(range(V), key=lambda j: logits[j])
        out.append(best)
        logits = step(best, pos)
        pos += 1
    return out, hidden_after_prompt, prompt_h, list(ids)


def mtp_draft_py(W, hidden, first, n_draft=3, hists=None, toks=None):
    """Match Session::mtp_draft (Qwen3.5 NextN: norm+concat+fc+attn/ffn+lm_head)."""
    h = hidden[:]
    token = first
    k_cache, v_cache = [], []
    drafts = []

    def attn_mlp(x, pos):
        xn = rmsnorm(x, W["mtp.attn_norm"])
        qg = gemv(W["mtp.wq"], xn)
        k = gemv(W["mtp.wk"], xn)
        v = gemv(W["mtp.wv"], xn)
        q = qg[: NQ * HD]
        gate = qg[NQ * HD :]
        qh = [rmsnorm(q[h * HD : (h + 1) * HD], W["mtp.qn"]) for h in range(NQ)]
        kh = [rmsnorm(k[h * HD : (h + 1) * HD], W["mtp.kn"]) for h in range(NKV)]
        vh = [v[h * HD : (h + 1) * HD] for h in range(NKV)]
        qh, kh = rope(qh, kh, pos)
        k_cache.append(kh)
        v_cache.append(vh)
        scale = 1.0 / math.sqrt(HD)
        rep = NQ // NKV
        o = []
        for hq in range(NQ):
            hkv = hq // rep
            scores = []
            m = -1e30
            for s in range(pos + 1):
                acc = sum(qh[hq][d] * k_cache[s][hkv][d] for d in range(HD)) * scale
                scores.append(acc)
                m = max(m, acc)
            z = sum(math.exp(s - m) for s in scores)
            oh = [0.0] * HD
            for s, sc in enumerate(scores):
                a = math.exp(sc - m) / z
                for d in range(HD):
                    oh[d] += a * v_cache[s][hkv][d]
            g = [sigmoid(gate[hq * HD + d]) for d in range(HD)]
            o.extend([oh[d] * g[d] for d in range(HD)])
        y = gemv(W["mtp.wo"], o)
        x = [x[d] + y[d] for d in range(H)]
        xn = rmsnorm(x, W["mtp.ffn_norm"])
        g = gemv(W["mtp.gate"], xn)
        u = gemv(W["mtp.up"], xn)
        hh = [silu(g[j]) * u[j] for j in range(INTER)]
        y = gemv(W["mtp.down"], hh)
        return [x[d] + y[d] for d in range(H)]

    def fuse_layer(href, tok, p):
        eh = rmsnorm(href, W["mtp.pre_h"])
        ee = rmsnorm(W["embed"][tok], W["mtp.pre_e"])
        cat = ee + eh  # NextN: concat(norm(embed), norm(hidden))
        # Stem only: skip the 1-layer decoder (matches Session::mtp_draft).
        return gemv(W["mtp.fc"], cat)

    # Stem, pre-final-norm residual. d0 last-hist; d1+ previous hist slot.
    n_hist = min(len(hists), len(toks)) if hists and toks else 0
    h_last = hists[n_hist - 1] if n_hist > 0 else hidden
    h_prev = hists[n_hist - 2] if n_hist > 1 else h_last
    h_d0 = h_last
    h_rest = h_prev
    if n_draft > 0:
        h = fuse_layer(h_d0, token, 0)
        hn = rmsnorm(h, W["mtp.norm"])
        logits = gemv(W["lm_head"], hn)
        token = max(range(V), key=lambda j: logits[j])
        drafts.append(token)
    while len(drafts) < n_draft:
        h = fuse_layer(h_rest, token, len(drafts))
        hn = rmsnorm(h, W["mtp.norm"])
        logits = gemv(W["lm_head"], hn)
        token = max(range(V), key=lambda j: logits[j])
        drafts.append(token)
    return drafts


def write_safetensors(path: Path, tensors: dict[str, tuple[list, list]]):
    """tensors: name -> (shape, flat f32 list)"""
    header = {}
    blobs = []
    off = 0
    for name, (shape, data) in tensors.items():
        nb = len(data) * 4
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + nb]}
        blobs.append(b"".join(struct.pack("<f", float(x)) for x in data))
        off += nb
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def flat(m):
    if not m:
        return []
    if isinstance(m[0], list):
        out = []
        for row in m:
            out.extend(row)
        return out
    return list(m)


def hf_tensors(W, kinds):
    t = {}
    t["model.language_model.embed_tokens.weight"] = ([V, H], flat(W["embed"]))
    t["lm_head.weight"] = ([V, H], flat(W["lm_head"]))
    t["model.language_model.norm.weight"] = ([H], W["final_norm"])
    t["model.visual.blocks.0.attn.qkv.weight"] = ([4, 4], [0.0] * 16)
    t["mtp.fc.weight"] = ([H, 2 * H], flat(W["mtp.fc"]))
    t["mtp.norm.weight"] = ([H], W["mtp.norm"])
    t["mtp.pre_fc_norm_hidden.weight"] = ([H], W["mtp.pre_h"])
    t["mtp.pre_fc_norm_embedding.weight"] = ([H], W["mtp.pre_e"])
    t["mtp.layers.0.input_layernorm.weight"] = ([H], W["mtp.attn_norm"])
    t["mtp.layers.0.post_attention_layernorm.weight"] = ([H], W["mtp.ffn_norm"])
    t["mtp.layers.0.self_attn.q_proj.weight"] = ([NQ * HD * 2, H], flat(W["mtp.wq"]))
    t["mtp.layers.0.self_attn.k_proj.weight"] = ([NKV * HD, H], flat(W["mtp.wk"]))
    t["mtp.layers.0.self_attn.v_proj.weight"] = ([NKV * HD, H], flat(W["mtp.wv"]))
    t["mtp.layers.0.self_attn.o_proj.weight"] = ([H, NQ * HD], flat(W["mtp.wo"]))
    t["mtp.layers.0.self_attn.q_norm.weight"] = ([HD], W["mtp.qn"])
    t["mtp.layers.0.self_attn.k_norm.weight"] = ([HD], W["mtp.kn"])
    t["mtp.layers.0.mlp.gate_proj.weight"] = ([INTER, H], flat(W["mtp.gate"]))
    t["mtp.layers.0.mlp.up_proj.weight"] = ([INTER, H], flat(W["mtp.up"]))
    t["mtp.layers.0.mlp.down_proj.weight"] = ([H, INTER], flat(W["mtp.down"]))
    for i, k in enumerate(kinds):
        p = f"L{i}"
        base = f"model.language_model.layers.{i}."
        t[base + "input_layernorm.weight"] = ([H], W[p + ".attn_norm"])
        t[base + "post_attention_layernorm.weight"] = ([H], W[p + ".ffn_norm"])
        t[base + "mlp.gate_proj.weight"] = ([INTER, H], flat(W[p + ".gate"]))
        t[base + "mlp.up_proj.weight"] = ([INTER, H], flat(W[p + ".up"]))
        t[base + "mlp.down_proj.weight"] = ([H, INTER], flat(W[p + ".down"]))
        if k == "D":
            qkv = NK * DK + NK * DK + NV * DV
            t[base + "linear_attn.in_proj_qkv.weight"] = ([qkv, H], flat(W[p + ".qkv"]))
            t[base + "linear_attn.in_proj_z.weight"] = ([NV * DV, H], flat(W[p + ".z"]))
            t[base + "linear_attn.out_proj.weight"] = ([H, NV * DV], flat(W[p + ".out"]))
            t[base + "linear_attn.in_proj_a.weight"] = ([NV, H], flat(W[p + ".a"]))
            t[base + "linear_attn.in_proj_b.weight"] = ([NV, H], flat(W[p + ".b"]))
            t[base + "linear_attn.A_log"] = ([NV], W[p + ".A_log"])
            t[base + "linear_attn.dt_bias"] = ([NV], W[p + ".dt"])
            t[base + "linear_attn.conv1d.weight"] = ([qkv, CONV_K], flat(W[p + ".conv"]))
            t[base + "linear_attn.norm.weight"] = ([NV * DV], W[p + ".gn"])
        else:
            t[base + "self_attn.q_proj.weight"] = ([NQ * HD * 2, H], flat(W[p + ".wq"]))
            t[base + "self_attn.k_proj.weight"] = ([NKV * HD, H], flat(W[p + ".wk"]))
            t[base + "self_attn.v_proj.weight"] = ([NKV * HD, H], flat(W[p + ".wv"]))
            t[base + "self_attn.o_proj.weight"] = ([H, NQ * HD], flat(W[p + ".wo"]))
            t[base + "self_attn.q_norm.weight"] = ([HD], W[p + ".qn"])
            t[base + "self_attn.k_norm.weight"] = ([HD], W[p + ".kn"])
    return t


def write_gguf(path: Path, tensors: dict, architecture="qwen35"):
    # name -> (shape, f32 list)
    def w_u8(f, v):
        f.write(struct.pack("<B", v))

    def w_u32(f, v):
        f.write(struct.pack("<I", v))

    def w_u64(f, v):
        f.write(struct.pack("<Q", v))

    def w_str(f, s):
        b = s.encode("utf-8")
        w_u64(f, len(b))
        f.write(b)

    items = list(tensors.items())
    kvs = [
        ("general.architecture", 8, architecture),
        ("qwen35.block_count", 4, L),
        ("qwen35.embedding_length", 4, H),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"GGUF")
        w_u32(f, 3)
        w_u64(f, len(items))
        w_u64(f, len(kvs))
        for key, typ, val in kvs:
            w_str(f, key)
            w_u32(f, typ)
            if typ == 8:
                w_str(f, val)
            else:
                w_u32(f, int(val))
        offset = 0
        metas = []
        for name, (shape, data) in items:
            nbytes = len(data) * 4
            metas.append((name, shape, 0, offset, data))
            offset += nbytes
        for name, shape, ggml, off, data in metas:
            w_str(f, name)
            w_u32(f, len(shape))
            for d in reversed(shape):
                w_u64(f, int(d))
            w_u32(f, ggml)  # F32
            w_u64(f, off)
        pos = f.tell()
        align = 32
        pad = (align - (pos % align)) % align
        f.write(b"\x00" * pad)
        for _, _, _, _, data in metas:
            f.write(b"".join(struct.pack("<f", float(x)) for x in data))


def gguf_names(kinds):
    t = {}
    return t


def to_gguf_tensors(W, kinds):
    t = {}
    t["token_embd.weight"] = ([V, H], flat(W["embed"]))
    t["output.weight"] = ([V, H], flat(W["lm_head"]))
    t["output_norm.weight"] = ([H], W["final_norm"])
    t["visual.dummy.weight"] = ([2, 2], [0.0] * 4)
    t["nextn.fc.weight"] = ([H, 2 * H], flat(W["mtp.fc"]))
    t["nextn.norm.weight"] = ([H], W["mtp.norm"])
    t["nextn.pre_fc_norm_hidden.weight"] = ([H], W["mtp.pre_h"])
    t["nextn.pre_fc_norm_embedding.weight"] = ([H], W["mtp.pre_e"])
    t["nextn.blk.0.attn_norm.weight"] = ([H], W["mtp.attn_norm"])
    t["nextn.blk.0.ffn_norm.weight"] = ([H], W["mtp.ffn_norm"])
    t["nextn.blk.0.attn_q.weight"] = ([NQ * HD * 2, H], flat(W["mtp.wq"]))
    t["nextn.blk.0.attn_k.weight"] = ([NKV * HD, H], flat(W["mtp.wk"]))
    t["nextn.blk.0.attn_v.weight"] = ([NKV * HD, H], flat(W["mtp.wv"]))
    t["nextn.blk.0.attn_output.weight"] = ([H, NQ * HD], flat(W["mtp.wo"]))
    t["nextn.blk.0.attn_q_norm.weight"] = ([HD], W["mtp.qn"])
    t["nextn.blk.0.attn_k_norm.weight"] = ([HD], W["mtp.kn"])
    t["nextn.blk.0.ffn_gate.weight"] = ([INTER, H], flat(W["mtp.gate"]))
    t["nextn.blk.0.ffn_up.weight"] = ([INTER, H], flat(W["mtp.up"]))
    t["nextn.blk.0.ffn_down.weight"] = ([H, INTER], flat(W["mtp.down"]))
    for i, k in enumerate(kinds):
        p = f"L{i}"
        b = f"blk.{i}."
        t[b + "attn_norm.weight"] = ([H], W[p + ".attn_norm"])
        t[b + "ffn_norm.weight"] = ([H], W[p + ".ffn_norm"])
        t[b + "ffn_gate.weight"] = ([INTER, H], flat(W[p + ".gate"]))
        t[b + "ffn_up.weight"] = ([INTER, H], flat(W[p + ".up"]))
        t[b + "ffn_down.weight"] = ([H, INTER], flat(W[p + ".down"]))
        if k == "D":
            qkv = NK * DK + NK * DK + NV * DV
            t[b + "ssm_in.weight"] = ([qkv, H], flat(W[p + ".qkv"]))
            t[b + "ssm_z.weight"] = ([NV * DV, H], flat(W[p + ".z"]))
            t[b + "ssm_out.weight"] = ([H, NV * DV], flat(W[p + ".out"]))
            t[b + "ssm_a.weight"] = ([NV, H], flat(W[p + ".a"]))
            t[b + "ssm_beta.weight"] = ([NV, H], flat(W[p + ".b"]))
            t[b + "ssm_a_log"] = ([NV], W[p + ".A_log"])
            t[b + "ssm_dt"] = ([NV], W[p + ".dt"])
            t[b + "ssm_conv1d.weight"] = ([qkv, CONV_K], flat(W[p + ".conv"]))
            t[b + "ssm_norm.weight"] = ([NV * DV], W[p + ".gn"])
        else:
            t[b + "attn_q.weight"] = ([NQ * HD * 2, H], flat(W[p + ".wq"]))
            t[b + "attn_k.weight"] = ([NKV * HD, H], flat(W[p + ".wk"]))
            t[b + "attn_v.weight"] = ([NKV * HD, H], flat(W[p + ".wv"]))
            t[b + "attn_output.weight"] = ([H, NQ * HD], flat(W[p + ".wo"]))
            t[b + "attn_q_norm.weight"] = ([HD], W[p + ".qn"])
            t[b + "attn_k_norm.weight"] = ([HD], W[p + ".kn"])
    return t


def write_bad_fp8(out: Path):
    d = out / "bad_fp8"
    d.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": H,
            "num_hidden_layers": L,
            "intermediate_size": INTER,
            "vocab_size": V,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
            "linear_num_key_heads": NK,
            "linear_num_value_heads": NV,
            "linear_key_head_dim": DK,
            "linear_value_head_dim": DV,
            "linear_conv_kernel_dim": CONV_K,
            "num_attention_heads": NQ,
            "num_key_value_heads": NKV,
            "head_dim": HD,
            "rms_norm_eps": EPS,
        },
    }
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    # F8_E4M3 without scale
    header = {
        "model.language_model.embed_tokens.weight": {
            "dtype": "F8_E4M3",
            "shape": [V, H],
            "data_offsets": [0, V * H],
        }
    }
    h = json.dumps(header, separators=(",", ":")).encode()
    with (d / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        f.write(b"\x00" * (V * H))


def write_bad_gguf(path: Path):
    # no DeltaNet tensors
    t = {"token_embd.weight": ([V, H], [0.0] * (V * H)), "blk.0.attn_q.weight": ([4, H], [0.0] * (4 * H))}
    write_gguf(path, t, architecture="llama")


def write_overquant_gguf(path: Path, W, kinds):
    # leftover A_log stored as Q8_0-marked via ggml type 8 but we write F32 bytes —
    # instead write a real GGUF with ggml type Q8_0 for ssm_a_log
    tensors = to_gguf_tensors(W, kinds)
    # We'll mark ssm_a_log as Q8_0 in a custom writer
    name = "blk.0.ssm_a_log"
    # build Q8_0 packing of zeros (NV=6, pad to 32)
    qs = bytes(2 + 32)  # one block
    # Use standard writer then it's F32. Custom:
    def w_u32(f, v):
        f.write(struct.pack("<I", v))

    def w_u64(f, v):
        f.write(struct.pack("<Q", v))

    def w_str(f, s):
        b = s.encode("utf-8")
        w_u64(f, len(b))
        f.write(b)

    items = [(name, [32], 8, qs)]  # ggml Q8_0
    with path.open("wb") as f:
        f.write(b"GGUF")
        w_u32(f, 3)
        w_u64(f, 2)
        w_u64(f, 1)
        w_str(f, "general.architecture")
        w_u32(f, 8)
        w_str(f, "qwen35")
        w_str(f, "token_embd.weight")
        w_u32(f, 2)
        w_u64(f, H)
        w_u64(f, V)
        w_u32(f, 0)
        w_u64(f, 0)
        w_str(f, name)
        w_u32(f, 1)
        w_u64(f, 32)
        w_u32(f, 8)
        w_u64(f, V * H * 4)
        pos = f.tell()
        pad = (32 - (pos % 32)) % 32
        f.write(b"\x00" * pad)
        f.write(b"\x00" * (V * H * 4))
        f.write(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    r = rng()
    W, kinds = make_weights(r)
    golden, h_prompt, prompt_h, prompt_toks = forward(W, kinds, PROMPT, N_NEW)
    mtp_ids = mtp_draft_py(W, h_prompt, golden[0], 3, prompt_h, prompt_toks)

    cfg = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": H,
            "num_hidden_layers": L,
            "intermediate_size": INTER,
            "vocab_size": V,
            "max_position_embeddings": 32768,
            "rms_norm_eps": EPS,
            "layer_types": [
                "linear_attention" if k == "D" else "full_attention" for k in kinds
            ],
            "linear_num_key_heads": NK,
            "linear_num_value_heads": NV,
            "linear_key_head_dim": DK,
            "linear_value_head_dim": DV,
            "linear_conv_kernel_dim": CONV_K,
            "num_attention_heads": NQ,
            "num_key_value_heads": NKV,
            "head_dim": HD,
            "partial_rotary_factor": 0.25,
            "rope_theta": THETA,
            "tie_word_embeddings": False,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
        },
    }
    hf = out / "tiny_hybrid"
    hf.mkdir(parents=True, exist_ok=True)
    (hf / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    write_safetensors(hf / "model.safetensors", hf_tensors(W, kinds))

    write_gguf(out / "tiny_hybrid.gguf", to_gguf_tensors(W, kinds))
    write_bad_fp8(out)
    write_bad_gguf(out / "bad_gguf.gguf")
    write_overquant_gguf(out / "bad_leftover.gguf", W, kinds)

    golden_obj = {
        "prompt_ids": PROMPT,
        "greedy_ids": golden,
        "n_new": N_NEW,
        "mtp_draft_ids": mtp_ids,
        "note": "stdlib reference matching RapidLLM HF-locked hybrid math + Qwen NextN MTP",
    }
    (hf / "golden.json").write_text(json.dumps(golden_obj, indent=2), encoding="utf-8")
    print("golden", golden)
    print("mtp_draft", mtp_ids)


if __name__ == "__main__":
    main()
