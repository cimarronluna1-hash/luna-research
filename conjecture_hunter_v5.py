#!/usr/bin/env python3
"""
CONJECTURE HUNTER v5.0 — FULL GPU (FIXED mod p²)
Cimarron Luna + Claude, March 16, 2026

FIX: K8-K10 used Fermat's theorem for mod p² inverse.
Fermat only works for PRIME modulus. p² is NOT prime.
Now uses Euler's theorem: a⁻¹ ≡ a^(p²-p-1) mod p²
since φ(p²) = p(p-1).

7 CUDA kernels:
  K1: C(2n,n)³/256ⁿ mod p     (D=-12, validated)
  K2: C(2n,n)³/64ⁿ mod p      (d=4)
  K3: C(2n,n)³/(-64)ⁿ mod p   (d=2)
  K4: Borwein/256ⁿ mod p
  K5: Domb/64ⁿ mod p           (FIXED recurrence)
  K6: Zagier z=1 mod p
  K7: Zagier z=-1 mod p

Plus 3 mod-p² kernels for carry chain:
  K8: C(2n,n)³/256ⁿ mod p²
  K9: Borwein/256ⁿ mod p²
  K10: Domb/64ⁿ mod p²

Run:
  cp ~/Downloads/conjecture_hunter_v4.py ~/luna_research/scripts/ && \
  cd ~/luna_research/scripts && python3 conjecture_hunter_v4.py
"""

import time, os, sys
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    print("ERROR: CuPy required"); sys.exit(1)

LIMIT = 500000
NCORES = cpu_count()

# =============================================================================
# CUDA HELPER: modpow and modinv as device functions
# =============================================================================

_DEVICE_FUNCS = r'''
__device__ long long d_modpow(long long base, long long exp, long long mod) {
    long long result = 1;
    base %= mod;
    if (base < 0) base += mod;
    while (exp > 0) {
        if (exp & 1) result = (__int128)result * base % mod;
        base = (__int128)base * base % mod;
        exp >>= 1;
    }
    return result;
}
__device__ long long d_modinv(long long a, long long p) {
    // Fermat: works only for PRIME p
    return d_modpow(a, p - 2, p);
}
__device__ long long d_modinv_p2(long long a, long long p) {
    // Euler: a^(phi(p²)-1) mod p² where phi(p²) = p(p-1)
    // So a^(-1) = a^(p²-p-1) mod p²
    long long M = p * p;
    long long exp = M - p - 1;  // p²-p-1 = p(p-1)-1
    return d_modpow(a, exp, M);
}
'''

# =============================================================================
# K1: C(2n,n)³/256ⁿ mod p (validated)
# =============================================================================
_K1 = cp.RawKernel(_DEVICE_FUNCS + r'''
extern "C" __global__
void k1_binom3_256(const long long* P, long long* R, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    long long p = P[idx];
    if (p < 7) { R[idx] = 0; return; }
    long long inv256 = d_modinv(256 % p, p);
    long long S = 0, t = 1;
    for (long long n = 0; n < p; n++) {
        S = (S + t) % p;
        if (n < p - 1) {
            long long n1 = n+1;
            long long ab = (__int128)(2*n+1) * (2*n+2) % p;
            long long ab3 = (__int128)ab * ab % p; ab3 = (__int128)ab3 * ab % p;
            long long d2 = (__int128)n1 * n1 % p;
            long long d6 = (__int128)d2 * d2 % p; d6 = (__int128)d6 * d2 % p;
            long long dinv = d_modinv(d6, p);
            t = (__int128)t * ab3 % p;
            t = (__int128)t * dinv % p;
            t = (__int128)t * inv256 % p;
        }
    }
    R[idx] = S;
}
''', 'k1_binom3_256', options=('--device-int128',))

# =============================================================================
# K2: C(2n,n)³/64ⁿ mod p
# =============================================================================
_K2 = cp.RawKernel(_DEVICE_FUNCS + r'''
extern "C" __global__
void k2_binom3_64(const long long* P, long long* R, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    long long p = P[idx];
    if (p < 7) { R[idx] = 0; return; }
    long long inv64 = d_modinv(64 % p, p);
    long long S = 0, t = 1;
    for (long long n = 0; n < p; n++) {
        S = (S + t) % p;
        if (n < p - 1) {
            long long n1 = n+1;
            long long ab = (__int128)(2*n+1) * (2*n+2) % p;
            long long ab3 = (__int128)ab * ab % p; ab3 = (__int128)ab3 * ab % p;
            long long d2 = (__int128)n1 * n1 % p;
            long long d6 = (__int128)d2 * d2 % p; d6 = (__int128)d6 * d2 % p;
            long long dinv = d_modinv(d6, p);
            t = (__int128)t * ab3 % p;
            t = (__int128)t * dinv % p;
            t = (__int128)t * inv64 % p;
        }
    }
    R[idx] = S;
}
''', 'k2_binom3_64', options=('--device-int128',))

# =============================================================================
# K3: C(2n,n)³/(-64)ⁿ mod p
# =============================================================================
_K3 = cp.RawKernel(_DEVICE_FUNCS + r'''
extern "C" __global__
void k3_binom3_n64(const long long* P, long long* R, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    long long p = P[idx];
    if (p < 7) { R[idx] = 0; return; }
    long long inv64 = d_modinv(64 % p, p);
    long long z = (p - inv64) % p;  // -1/64 mod p
    long long S = 0, t = 1;
    for (long long n = 0; n < p; n++) {
        S = (S + t) % p;
        if (n < p - 1) {
            long long n1 = n+1;
            long long ab = (__int128)(2*n+1) * (2*n+2) % p;
            long long ab3 = (__int128)ab * ab % p; ab3 = (__int128)ab3 * ab % p;
            long long d2 = (__int128)n1 * n1 % p;
            long long d6 = (__int128)d2 * d2 % p; d6 = (__int128)d6 * d2 % p;
            long long dinv = d_modinv(d6, p);
            t = (__int128)t * ab3 % p;
            t = (__int128)t * dinv % p;
            t = (__int128)t * z % p;
        }
    }
    R[idx] = S;
}
''', 'k3_binom3_n64', options=('--device-int128',))

# =============================================================================
# K4: Borwein C(2n,n)²C(4n,2n)/256ⁿ mod p
# =============================================================================
_K4 = cp.RawKernel(_DEVICE_FUNCS + r'''
extern "C" __global__
void k4_borwein(const long long* P, long long* R, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    long long p = P[idx];
    if (p < 7) { R[idx] = 0; return; }
    long long inv256 = d_modinv(256 % p, p);
    long long S = 0, t = 1;
    for (long long n = 0; n < p; n++) {
        S = (S + t) % p;
        if (n < p - 1) {
            long long n1 = n+1;
            // num = (4n+1)(4n+2)(4n+3)(4n+4)
            long long nm = (__int128)(4*n+1) * (4*n+2) % p;
            nm = (__int128)nm * ((4*n+3) % p) % p;
            nm = (__int128)nm * ((4*n+4) % p) % p;
            // den = (n+1)^4
            long long d2 = (__int128)n1 * n1 % p;
            long long d4 = (__int128)d2 * d2 % p;
            long long dinv = d_modinv(d4, p);
            t = (__int128)t * nm % p;
            t = (__int128)t * dinv % p;
            t = (__int128)t * inv256 % p;
        }
    }
    R[idx] = S;
}
''', 'k4_borwein', options=('--device-int128',))

# =============================================================================
# K5: Domb at z=1/64 mod p (FIXED recurrence)
# =============================================================================
_K5 = cp.RawKernel(_DEVICE_FUNCS + r'''
extern "C" __global__
void k5_domb(const long long* P, long long* R, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    long long p = P[idx];
    if (p < 7) { R[idx] = 0; return; }
    long long iz64 = d_modinv(64 % p, p);
    long long ap = 0, ac = 1;
    long long S = 0, zp = 1;
    for (long long n = 0; n < p; n++) {
        S = (S + (__int128)ac * zp % p) % p;
        if (n < p - 1) {
            zp = (__int128)zp * iz64 % p;
            long long n1 = n + 1;
            // c1 = 2(2n+1)(5n²+5n+2) — FIXED
            long long inner = ((__int128)5*n*n + 5*n + 2) % p;
            long long c1 = (__int128)2 * ((2*n+1) % p) % p;
            c1 = (__int128)c1 * inner % p;
            // c2 = 64n³
            long long c2 = (__int128)64 * n % p;
            c2 = (__int128)c2 * n % p;
            c2 = (__int128)c2 * n % p;
            long long d3 = (__int128)n1 * n1 % p;
            d3 = (__int128)d3 * n1 % p;
            long long di = d_modinv(d3, p);
            long long an = ((__int128)c1 * ac % p - (__int128)c2 * ap % p + p) % p;
            an = (__int128)an * di % p;
            ap = ac; ac = an;
        }
    }
    R[idx] = S;
}
''', 'k5_domb', options=('--device-int128',))

# =============================================================================
# K6: Zagier z=1 mod p
# =============================================================================
_K6 = cp.RawKernel(_DEVICE_FUNCS + r'''
extern "C" __global__
void k6_zagier1(const long long* P, long long* R, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    long long p = P[idx];
    if (p < 7) { R[idx] = 0; return; }
    long long ap = 0, ac = 1;
    long long S = 0;
    for (long long n = 0; n < p; n++) {
        S = (S + ac) % p;
        if (n < p - 1) {
            long long n1 = n + 1;
            long long inner = ((__int128)17*n*n + 17*n + 5) % p;
            long long c1 = (__int128)(2*n+1) * inner % p;
            long long c2 = (__int128)n * n % p; c2 = (__int128)c2 * n % p;
            long long d3 = (__int128)n1 * n1 % p; d3 = (__int128)d3 * n1 % p;
            long long di = d_modinv(d3, p);
            long long an = ((__int128)c1 * ac % p - (__int128)c2 * ap % p + p) % p;
            an = (__int128)an * di % p;
            ap = ac; ac = an;
        }
    }
    R[idx] = S;
}
''', 'k6_zagier1', options=('--device-int128',))

# =============================================================================
# K7: Zagier z=-1 mod p
# =============================================================================
_K7 = cp.RawKernel(_DEVICE_FUNCS + r'''
extern "C" __global__
void k7_zagiern1(const long long* P, long long* R, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    long long p = P[idx];
    if (p < 7) { R[idx] = 0; return; }
    long long ap = 0, ac = 1;
    long long S = 0, zp = 1;
    long long zneg = p - 1;
    for (long long n = 0; n < p; n++) {
        S = (S + (__int128)ac * zp % p) % p;
        if (n < p - 1) {
            zp = (__int128)zp * zneg % p;
            long long n1 = n + 1;
            long long inner = ((__int128)17*n*n + 17*n + 5) % p;
            long long c1 = (__int128)(2*n+1) * inner % p;
            long long c2 = (__int128)n * n % p; c2 = (__int128)c2 * n % p;
            long long d3 = (__int128)n1 * n1 % p; d3 = (__int128)d3 * n1 % p;
            long long di = d_modinv(d3, p);
            long long an = ((__int128)c1 * ac % p - (__int128)c2 * ap % p + p) % p;
            an = (__int128)an * di % p;
            ap = ac; ac = an;
        }
    }
    R[idx] = S;
}
''', 'k7_zagiern1', options=('--device-int128',))

# =============================================================================
# K8-K10: mod p² kernels for carry chain
# p ≤ 500K → p² ≤ 2.5×10¹¹ → fits int64, use __int128 for products
# =============================================================================

_K8 = cp.RawKernel(_DEVICE_FUNCS + r'''
extern "C" __global__
void k8_binom3_256_p2(const long long* P, long long* R, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    long long p = P[idx];
    if (p < 7) { R[idx] = 0; return; }
    long long M = p * p;
    long long inv256 = d_modinv_p2(256 % M, p);
    long long S = 0, t = 1;
    for (long long n = 0; n < p; n++) {
        S = (S + t) % M;
        if (n < p - 1) {
            long long n1 = n+1;
            long long ab = (__int128)(2*n+1) * (2*n+2) % M;
            long long ab3 = (__int128)ab * ab % M; ab3 = (__int128)ab3 * ab % M;
            long long d2 = (__int128)n1 * n1 % M;
            long long d6 = (__int128)d2 * d2 % M; d6 = (__int128)d6 * d2 % M;
            long long dinv = d_modinv_p2(d6, p);
            t = (__int128)t * ab3 % M;
            t = (__int128)t * dinv % M;
            t = (__int128)t * inv256 % M;
        }
    }
    R[idx] = S;
}
''', 'k8_binom3_256_p2', options=('--device-int128',))

_K9 = cp.RawKernel(_DEVICE_FUNCS + r'''
extern "C" __global__
void k9_borwein_p2(const long long* P, long long* R, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    long long p = P[idx];
    if (p < 7) { R[idx] = 0; return; }
    long long M = p * p;
    long long inv256 = d_modinv_p2(256 % M, p);
    long long S = 0, t = 1;
    for (long long n = 0; n < p; n++) {
        S = (S + t) % M;
        if (n < p - 1) {
            long long n1 = n+1;
            long long nm = (__int128)(4*n+1) * (4*n+2) % M;
            nm = (__int128)nm * ((4*n+3) % M) % M;
            nm = (__int128)nm * ((4*n+4) % M) % M;
            long long d2 = (__int128)n1 * n1 % M;
            long long d4 = (__int128)d2 * d2 % M;
            long long dinv = d_modinv_p2(d4, p);
            t = (__int128)t * nm % M;
            t = (__int128)t * dinv % M;
            t = (__int128)t * inv256 % M;
        }
    }
    R[idx] = S;
}
''', 'k9_borwein_p2', options=('--device-int128',))

_K10 = cp.RawKernel(_DEVICE_FUNCS + r'''
extern "C" __global__
void k10_domb_p2(const long long* P, long long* R, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    long long p = P[idx];
    if (p < 7) { R[idx] = 0; return; }
    long long M = p * p;
    long long iz64 = d_modinv_p2(64 % M, p);
    long long ap = 0, ac = 1;
    long long S = 0, zp = 1;
    for (long long n = 0; n < p; n++) {
        S = (S + (__int128)ac * zp % M) % M;
        if (n < p - 1) {
            zp = (__int128)zp * iz64 % M;
            long long n1 = n + 1;
            long long inner = ((__int128)5*n*n + 5*n + 2) % M;
            long long c1 = (__int128)2 * ((2*n+1) % M) % M;
            c1 = (__int128)c1 * inner % M;
            long long c2 = (__int128)64 * n % M;
            c2 = (__int128)c2 * n % M;
            c2 = (__int128)c2 * n % M;
            long long d3 = (__int128)n1 * n1 % M;
            d3 = (__int128)d3 * n1 % M;
            long long di = d_modinv_p2(d3, p);
            long long an = ((__int128)c1 * ac % M - (__int128)c2 * ap % M + M) % M;
            an = (__int128)an * di % M;
            ap = ac; ac = an;
        }
    }
    R[idx] = S;
}
''', 'k10_domb_p2', options=('--device-int128',))


# =============================================================================
# GPU DISPATCH
# =============================================================================

def gpu_run(kernel, primes_np):
    n = len(primes_np)
    p_gpu = cp.asarray(primes_np, dtype=cp.int64)
    r_gpu = cp.zeros(n, dtype=cp.int64)
    threads, blocks = 256, (n + 255) // 256
    kernel((blocks,), (threads,), (p_gpu, r_gpu, n))
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(r_gpu)


# =============================================================================
# CORNACCHIA — parallel on CPU
# =============================================================================

def _egcd(a, b):
    if a == 0: return b, 0, 1
    g, x, y = _egcd(b % a, a)
    return g, y - (b // a) * x, x

def cornacchia(p, d):
    negd = (-d) % p
    if negd == 0: return None
    if pow(negd, (p-1)//2, p) != 1: return None
    Q, S = p - 1, 0
    while Q % 2 == 0: Q //= 2; S += 1
    z = 2
    while pow(z, (p-1)//2, p) != p - 1:
        z += 1
        if z > 10000: return None
    Mv, c, t, R = S, pow(z, Q, p), pow(negd, Q, p), pow(negd, (Q+1)//2, p)
    while True:
        if t == 1: break
        i, temp = 1, (t*t) % p
        while temp != 1:
            temp = (temp*temp) % p; i += 1
            if i > S: return None
        b2 = pow(c, 1 << (Mv-i-1), p)
        Mv, c, t, R = i, (b2*b2)%p, (t*b2*b2)%p, (R*b2)%p
    r = R
    for r_try in [r, p - r]:
        r0, r1 = p, r_try
        lim = int(p ** 0.5)
        while r1 > lim: r0, r1 = r1, r0 % r1
        a_c = r1
        rem = p - a_c * a_c
        if rem <= 0 or rem % d != 0: continue
        b_sq = rem // d
        b_c = int(b_sq ** 0.5 + 0.5)
        if b_c * b_c != b_sq: continue
        if a_c * a_c + d * b_c * b_c != p: continue
        return (a_c, b_c)
    return None

def corn_worker(args):
    p, disc_list = args
    result = {'p': p}
    for d in disc_list:
        c = cornacchia(p, d)
        if c: result[d] = c
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0_total = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  CONJECTURE HUNTER v5.0 — FULL GPU (FIXED mod p²)        ║")
    print("║  Cimarron Luna + Claude, March 16, 2026                    ║")
    print(f"║  10 CUDA kernels on RTX 3090 + {NCORES}-core Cornacchia{' ':>{21-len(str(NCORES))}}║")
    print(f"║  Limit: {LIMIT:,}  ({LIMIT//1000}K primes){' ':>{36-len(f'{LIMIT:,}')}}║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Sieve
    is_p = bytearray([1]) * (LIMIT + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_p[i]:
            for j in range(i*i, LIMIT + 1, i): is_p[j] = 0
    primes = [i for i in range(7, LIMIT + 1) if is_p[i]]
    primes_np = np.array(primes, dtype=np.int64)
    print(f"  Primes: {len(primes):,}\n")

    # =====================================================================
    # STEP 1: ALL GPU KERNELS
    # =====================================================================
    print("=" * 70)
    print("STEP 1: GPU COMPUTE — 10 kernels")
    print("=" * 70)

    kernels = [
        ("K1  C(2n,n)³/256ⁿ mod p",   _K1),
        ("K2  C(2n,n)³/64ⁿ mod p",    _K2),
        ("K3  C(2n,n)³/(-64)ⁿ mod p", _K3),
        ("K4  Borwein/256ⁿ mod p",     _K4),
        ("K5  Domb/64ⁿ mod p",         _K5),
        ("K6  Zagier z=1 mod p",       _K6),
        ("K7  Zagier z=-1 mod p",      _K7),
        ("K8  C(2n,n)³/256ⁿ mod p²",  _K8),
        ("K9  Borwein/256ⁿ mod p²",    _K9),
        ("K10 Domb/64ⁿ mod p²",        _K10),
    ]

    gpu_data = {}
    t0_gpu = time.time()
    for name, kernel in kernels:
        t0 = time.time()
        gpu_data[name[:3].strip()] = gpu_run(kernel, primes_np)
        elapsed = time.time() - t0
        print(f"  {name}: {elapsed:.1f}s ({len(primes)/elapsed:,.0f} p/s)")
    gpu_total = time.time() - t0_gpu
    print(f"\n  Total GPU: {gpu_total:.1f}s")

    # =====================================================================
    # STEP 2: CORNACCHIA — 24 cores
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"STEP 2: CORNACCHIA — {NCORES} cores")
    print("=" * 70)

    disc_list = [2, 3, 4, 15]
    t0 = time.time()
    with Pool(NCORES) as pool:
        corn_results = pool.map(corn_worker,
            [(p, disc_list) for p in primes], chunksize=256)
    corn_time = time.time() - t0
    print(f"  {len(primes):,} primes × {len(disc_list)} discriminants in {corn_time:.1f}s")

    # Index by prime
    corn_by_p = {r['p']: r for r in corn_results}

    # =====================================================================
    # STEP 3: IDENTITY VERIFICATION
    # =====================================================================
    print(f"\n{'='*70}")
    print("STEP 3: IDENTITY VERIFICATION")
    print("=" * 70)

    checks = [
        ("C(2n,n)³/256ⁿ d=3 ε",   'K1', 3, True),
        ("C(2n,n)³/64ⁿ d=4 ε",    'K2', 4, True),
        ("C(2n,n)³/(-64)ⁿ d=2 ε", 'K3', 2, True),
        ("Borwein/256ⁿ d=2 no ε",  'K4', 2, False),
        ("Domb/64ⁿ d=15 no ε",     'K5', 15, False),
        ("Zagier z=1 d=2 no ε",    'K6', 2, False),
        ("Zagier z=-1 d=3 no ε",   'K7', 3, False),
    ]

    for name, key, d, use_eps in checks:
        arr = gpu_data[key]
        sm, st, iv, it = 0, 0, 0, 0
        for i, p in enumerate(primes):
            S0 = int(arr[i])
            eps = 1 if ((p-1)//2) % 2 == 0 else -1
            cr = corn_by_p.get(p, {})
            corn = cr.get(d)
            if corn is None:
                it += 1
                if S0 == 0: iv += 1
            else:
                st += 1
                a, b = corn
                H = 2*(a*a - d*b*b)
                target = (eps*H) % p if use_eps else H % p
                if S0 == target: sm += 1
        status = '★★★ PERFECT' if sm == st and st > 0 else f'{sm}/{st}'
        print(f"  {name:>35s}: {status}  inert:{iv}/{it}")

    # =====================================================================
    # STEP 4: CARRY CHAIN b₁ UNIVERSALITY
    # =====================================================================
    print(f"\n{'='*70}")
    print("STEP 4: CARRY CHAIN b₁ — THE DISCOVERY")
    print("=" * 70)

    carry = [
        ("D=-12 C(2n,n)³/256ⁿ", 'K8', 3, True),
        ("Borwein/256ⁿ",         'K9', 2, False),
        ("Domb/64ⁿ",             'K10', 15, False),
    ]

    for name, key, d, use_eps in carry:
        arr = gpu_data[key]
        b1_mod6 = Counter()
        b1_mod2 = Counter()
        n_tested = 0
        for i, p in enumerate(primes):
            S0 = int(arr[i])
            eps = 1 if ((p-1)//2) % 2 == 0 else -1
            cr = corn_by_p.get(p, {})
            corn = cr.get(d)
            if corn is None: continue
            a, b = corn
            H = 2*(a*a - d*b*b)
            ap = (eps * H) if use_eps else H
            M2 = p * p
            diff = (S0 - ap) % M2
            if diff % p != 0: continue
            b1 = (diff // p) % p
            b1_mod6[b1 % 6] += 1
            b1_mod2[b1 % 2] += 1
            n_tested += 1

        print(f"\n  {name} (d={d}):")
        print(f"    Tested: {n_tested:,}")
        print(f"    b₁ mod 6: {dict(sorted(b1_mod6.items()))}")
        print(f"    b₁ mod 2: {dict(sorted(b1_mod2.items()))}")
        if len(b1_mod6) == 1 and n_tested > 100:
            val = list(b1_mod6.keys())[0]
            print(f"    ★★★★★ b₁ ≡ {val} mod 6 for ALL {n_tested:,} primes — UNIVERSAL THEOREM")
        elif len(b1_mod6) <= 3 and n_tested > 100:
            print(f"    ★★ Discrete structure detected")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    elapsed = time.time() - t0_total
    print(f"\n{'='*70}")
    print(f"CONJECTURE HUNTER v5.0 COMPLETE")
    print(f"  GPU: {gpu_total:.1f}s  Cornacchia: {corn_time:.1f}s  Total: {elapsed:.1f}s")
    print(f"  Primes: {len(primes):,}  Limit: {LIMIT:,}")
    print(f"{'='*70}")

    # Save
    outfile = os.path.expanduser('~/luna_research/results/conjecture_hunter_v5.txt')
    try:
        with open(outfile, 'w') as f:
            f.write(f"Conjecture Hunter v5 — FULL GPU, FIXED mod p²\n")
            f.write(f"Primes: {len(primes):,}, Time: {elapsed:.1f}s\n")
            f.write(f"GPU: {gpu_total:.1f}s, Cornacchia: {corn_time:.1f}s\n")
        print(f"\n  Saved: {outfile}")
    except: pass


if __name__ == '__main__':
    main()
