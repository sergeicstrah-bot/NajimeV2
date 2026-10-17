import numpy as np
from numba import njit
import itertools
from collections import deque

DIM   = 6
DIMS  = (5, 5, 5, 4, 3, 2)
N_CELLS = 5*5*5*4*3*2
STEP    = 5.0

STRIDES = np.zeros(DIM, dtype=np.int64)
_s = 1
for i in range(DIM-1, -1, -1):
    STRIDES[i] = _s
    _s *= DIMS[i]
DIMS_ARR = np.array(DIMS, dtype=np.int64)

ALL_OFFS = np.array(
    [o for o in itertools.product((-1,0,1), repeat=DIM) if any(o)],
    dtype=np.int8,
)
NOFF = len(ALL_OFFS)

def _build_order_idx():
    keys = []
    for i, o in enumerate(ALL_OFFS):
        nz = [(j, int(v)) for j, v in enumerate(o) if v != 0]
        keys.append((len(nz), nz[0][0], nz[0][1], tuple(int(v) for v in o)))
    return np.array(sorted(range(NOFF), key=lambda i: keys[i]), dtype=np.int32)

ORDER_IDX = _build_order_idx()


@njit
def build_neighbor_table(n_cells, noff, all_offs, dims, strides):
    table = np.empty((n_cells, noff), dtype=np.int32)
    for i in range(n_cells):
        for j in range(noff):
            o = all_offs[j]
            c0 = (i // strides[0]) % dims[0]
            c1 = (i // strides[1]) % dims[1]
            c2 = (i // strides[2]) % dims[2]
            c3 = (i // strides[3]) % dims[3]
            c4 = (i // strides[4]) % dims[4]
            c5 = (i // strides[5]) % dims[5]
            table[i, j] = (((c0+o[0])%dims[0])*strides[0]
                         + ((c1+o[1])%dims[1])*strides[1]
                         + ((c2+o[2])%dims[2])*strides[2]
                         + ((c3+o[3])%dims[3])*strides[3]
                         + ((c4+o[4])%dims[4])*strides[4]
                         + ((c5+o[5])%dims[5])*strides[5])
    return table


@njit
def step_kernel(potential, conn_idx, conn_count, next_addr,
                spiking_buf, delta_buf, worst_ratio_buf,
                neighbor_table, order_idx,
                step_thresh, noff):
    n = potential.shape[0]

    # 1) спайк = локальный максимум среди целей
    nsp = 0
    for i in range(n):
        pi = potential[i]
        if pi <= 0.0: continue
        cnt = conn_count[i]
        is_max = True
        for j in range(cnt):
            dst = neighbor_table[i, conn_idx[i, j]]
            if potential[dst] >= pi:
                is_max = False
                break
        if is_max:
            spiking_buf[nsp] = i
            nsp += 1

    # 2) связка строже спайка: i > 2*dst (без умножения)
    for k in range(nsp):
        c = spiking_buf[k]
        if next_addr[c] >= noff: continue
        off_idx = order_idx[next_addr[c]]
        dst = neighbor_table[c, off_idx]
        pd = potential[dst]
        if potential[c] > pd + pd:
            conn_idx[c, conn_count[c]] = off_idx
            conn_count[c] += 1
        next_addr[c] += 1

    # 3) передача: C = (pi - min_target) / cnt
    for i in range(n):
        delta_buf[i] = 0.0
    for k in range(nsp):
        c = spiking_buf[k]
        cnt = conn_count[c]
        if cnt == 0: continue
        pi = potential[c]

        min_pd = pi
        for j in range(cnt):
            pd = potential[neighbor_table[c, conn_idx[c, j]]]
            if pd < min_pd:
                min_pd = pd

        C = (pi - min_pd) / cnt

        total = 0.0
        for j in range(cnt):
            pd = potential[neighbor_table[c, conn_idx[c, j]]]
            total += (pi - pd) + C
        if total <= 0.0: continue

        potential[c] = 0.0
        min_share = 1e300
        max_share = -1e300
        for j in range(cnt):
            dst = neighbor_table[c, conn_idx[c, j]]
            pd = potential[dst]
            share = pi * ((pi - pd) + C) / total
            delta_buf[dst] += share
            if share < min_share: min_share = share
            if share > max_share: max_share = share

        if cnt > 1 and max_share > 0.0:
            ratio = min_share / max_share
            if ratio < worst_ratio_buf[cnt]:
                worst_ratio_buf[cnt] = ratio

    for i in range(n):
        if delta_buf[i] != 0.0:
            potential[i] += delta_buf[i]

    # 4) утечка
    for i in range(n):
        v = potential[i] - step_thresh
        potential[i] = v if v > 0.0 else 0.0

    return nsp


class World:
    def __init__(self, input_coords=(0, 0, 0, 0, 0, 0)):
        self.potential  = np.zeros(N_CELLS, dtype=np.float64)
        self.conn_idx   = np.full((N_CELLS, NOFF), -1, dtype=np.int32)
        self.conn_count = np.zeros(N_CELLS, dtype=np.int32)
        self.next_addr  = np.zeros(N_CELLS, dtype=np.int32)
        self.neighbor_table = build_neighbor_table(N_CELLS, NOFF, ALL_OFFS, DIMS_ARR, STRIDES)
        self.spiking_buf = np.empty(N_CELLS, dtype=np.int32)
        self.delta_buf   = np.empty(N_CELLS, dtype=np.float64)
        self.worst_ratio_buf = np.ones(NOFF + 1, dtype=np.float64)
        self.tick = 0
        self.last_nsp = 0
        self.input_idx = self._coords_to_idx(input_coords)

    def _coords_to_idx(self, coords):
        idx = 0
        for i in range(DIM):
            idx += (coords[i] % DIMS[i]) * STRIDES[i]
        return idx

    def inject(self, amount, coords=None):
        idx = self.input_idx if coords is None else self._coords_to_idx(coords)
        self.potential[idx] += amount

    def step(self):
        self.last_nsp = step_kernel(
            self.potential, self.conn_idx, self.conn_count, self.next_addr,
            self.spiking_buf, self.delta_buf, self.worst_ratio_buf,
            self.neighbor_table, ORDER_IDX, STEP, NOFF,
        )
        self.tick += 1


def run_sim(inject_amount, ticks=50, inject_ticks=1):
    w = World()
    for t in range(ticks):
        if t < inject_ticks:
            w.inject(inject_amount)
        w.step()
    return w


def bfs_depth(w):
    depth = np.full(N_CELLS, -1, dtype=np.int32)
    depth[w.input_idx] = 0
    q = deque([w.input_idx])
    while q:
        c = q.popleft()
        for j in range(w.conn_count[c]):
            dst = w.neighbor_table[c, w.conn_idx[c, j]]
            if depth[dst] == -1:
                depth[dst] = depth[c] + 1
                q.append(dst)
    return depth


def measure(w):
    conns = int(np.sum(w.conn_count))
    max_cell = int(w.conn_count.max())
    cells_with = int((w.conn_count > 0).sum())
    depth = bfs_depth(w)
    max_depth = int(depth.max())
    active = int((w.potential > 0).sum())
    ratios = w.worst_ratio_buf[2:]
    mask = ratios < 1.0
    worst_ratio = float(ratios[mask].min()) if mask.any() else 1.0
    return dict(conns=conns, max_cell=max_cell, cells_with=cells_with,
                bfs_depth=max_depth, active=active, worst_ratio=worst_ratio)


def P(name, passed, info):
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {info}")


if __name__ == '__main__':
    # T1 non-degeneracy
    w = run_sim(20.0, ticks=50)
    m = measure(w)
    P("non-degeneracy A=20", m['conns'] > 0,
      f"conns={m['conns']} bfs_depth={m['bfs_depth']}")

    # T2 linear scaling
    conns_by_A = {}
    for A in (5, 10, 15, 20):
        m = measure(run_sim(float(A), ticks=50))
        conns_by_A[A] = m['conns']
        print(f"  A={A:3d}  conns={m['conns']:3d}  expected={A/5:.1f}  "
              f"bfs_depth={m['bfs_depth']}  max_cell={m['max_cell']}")
    ok = all(abs(conns_by_A[A] - A/5) <= 1 for A in (5, 10, 15, 20))
    P("linear scaling A<=20", ok, f"conns={conns_by_A} expected≈A/5")

    # T3 non-explosion
    m = measure(run_sim(160.0, ticks=200))
    P("non-explosion A=160", m['max_cell'] < 100,
      f"max_cell={m['max_cell']} conns={m['conns']} bfs_depth={m['bfs_depth']}")

    # T4 non-topK
    m = measure(run_sim(80.0, ticks=200))
    P("non-topK A=80", m['worst_ratio'] > 0.03,
      f"worst_ratio={m['worst_ratio']:.3f} (bound≈1/(cnt+1))")

    # T5 distributed graph
    m = measure(run_sim(80.0, ticks=200))
    P("distributed A=80", m['cells_with'] >= 5,
      f"cells_with_conns={m['cells_with']} max_cell={m['max_cell']}")

    # T6 no N^2
    m = measure(run_sim(80.0, ticks=200))
    P("no N^2 A=80", m['conns'] < N_CELLS // 10,
      f"conns={m['conns']} < {N_CELLS // 10}")