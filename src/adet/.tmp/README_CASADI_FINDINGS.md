# CasADi OpenMP Issue - Investigation Summary

## Problem
`rootfinder('nlpsol')` with `.map(n, 'openmp')` crashes with memory corruption, even though Ipopt 3.14.11 is thread-safe.

## Root Cause
**Bug in CasADi's `rootfinder` wrapper**: The wrapper doesn't properly allocate thread-local workspace memory when using OpenMP, causing buffer overlaps between threads.

**Evidence:**
```
Memory corruption detected for input lam_f.
arg[2] 0x2e002398-0x2e0023a0 intersects with w 0x2e002380-0x2e0023e0
```

## What Works vs What Doesn't

### ✅ Works
- `rootfinder('nlpsol')` + serial mapping
- `rootfinder('newton')` + OpenMP mapping
- `rootfinder('kinsol')` + OpenMP mapping
- `nlpsol` directly + OpenMP mapping (but no speedup, slight slowdown)
- Simple MX functions + OpenMP mapping

### ❌ Doesn't Work
- `rootfinder('nlpsol')` + OpenMP mapping → **CRASHES**
- `rootfinder('nlpsol')` + thread mapping → **CRASHES**

## Solutions

### Option 1: Use Serial Mapping (Recommended)
```python
rootfinder = cs.rootfinder('rf', 'nlpsol', problem, opts)
rootfinder_mapped = rootfinder.map(num_span, 'serial')  # Change to 'serial'
```
- ✅ Robust, no crashes
- ✅ Same results as before
- ❌ No parallelization (~21s for 10k solves)

### Option 2: Use nlpsol Directly
```python
# Reformulate g(x) = 0 as NLP with equality constraint
nlp = {'x': x, 'f': 0, 'g': g_expr, 'p': p}
solver = cs.nlpsol('solver', 'ipopt', nlp, opts)
solver_mapped = solver.map(num_span, 'openmp')
sol = solver_mapped(x0=x0_vals, p=p_vals, lbg=[0]*n, ubg=[0]*n)
```
- ✅ OpenMP works (no crash)
- ✅ More flexible (can add bounds, objectives)
- ❌ Returns dict instead of vector
- ❌ Slightly slower due to OpenMP overhead (0.86x speedup = slowdown)

### Option 3: Switch to Newton Method
```python
rootfinder = cs.rootfinder('rf', 'newton', problem, opts)
rootfinder_mapped = rootfinder.map(num_span, 'openmp')
```
- ✅ OpenMP works perfectly
- ✅ ~130x faster than Ipopt for simple problems
- ❌ Less robust, can fail on difficult problems

## Key Differences: rootfinder vs nlpsol

| Feature | rootfinder | nlpsol |
|---------|-----------|--------|
| Purpose | Solving g(x) = 0 | General NLP optimization |
| API | Clean, direct | Need to reformulate as constraints |
| Returns | Vector (DM) | Dict with x, f, g, multipliers |
| OpenMP with Ipopt | ❌ Crashes | ✅ Works (but slow) |
| OpenMP with Newton | ✅ Works | N/A |

## Environment Setup Required

Add to your `~/.bashrc` or `~/.zshrc`:
```bash
export LD_LIBRARY_PATH=/home/francesco/repos/casadi/build/lib:$LD_LIBRARY_PATH
```

Or use the wrapper script:
```bash
./run_casadi.sh
```

## Is This a Build Issue?

**No.** Investigation showed:
- ✅ CasADi built with `WITH_OPENMP=ON`, `WITH_THREAD=ON`, `WITH_THREADSAFE_SYMBOLICS=ON`
- ✅ Ipopt 3.14.11 with MUMPS 5.4.1 (thread-safe with mutex protection)
- ✅ Both link to libgomp correctly
- ✅ OpenMP works with other CasADi functions
- ❌ Bug is specific to `rootfinder` wrapper's memory management

## Recommendation

For robust Ipopt-based rootfinding with your current setup:
1. **Best:** Use `rootfinder` with `'serial'` mapping
2. **Alternative:** Use `nlpsol` directly (works with OpenMP but no speedup)
3. **If applicable:** Use `'newton'` backend for simple problems (fast + OpenMP)

Consider reporting this as a bug to the CasADi team: https://github.com/casadi/casadi/issues
