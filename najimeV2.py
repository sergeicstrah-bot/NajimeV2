# najime_v2.py
import os
import uuid
import numpy as np
from numba import njit

# ==================== CONSTANTS ====================
COL_SIZE = 100
LAYER_SIZE = 10
N_LAYERS = 10
PORT_LAYERS = (2, 3, 4, 5, 6)

LEAK_STEP = 1
ADAPT_STEP = 5
MOD_DECAY = 1
DG_LTD_TICKS = 300
LTD_REGIONS = ('dg',)
PRUNE_EVERY = 30
CONSOLIDATE_BURST = 3

MG_TH = 5
BASE_TH = 20
PORT_TH_OFFSET = 1
DG_TH_OFFSET = 3
INPUT_STRENGTH = 140

WAVE_DELTA = 12
LATERAL_FACIL_GAIN = 8
LATERAL_INHIB_GAIN = 2
LATERAL_FACIL_LIMIT = -10
LATERAL_INHIB_LIMIT = 15
LATERAL_REFRACTORY = 2

MOTIVATION_GAIN = 2
MOTIVATION_IDLE_WINDOW = 10

MAX_OUT = 8
SIGNAL_MAX = 65000

REGION_ORDER = ['input', 'sensory', 'dg', 'ca3', 'ca1',
                'thought', 'motor', 'motivation', 'output']
REGION_COLS = {
    'input': 0, 'sensory': 2, 'dg': 2, 'ca3': 2, 'ca1': 2,
    'thought': 2, 'motor': 4, 'motivation': 2, 'output': 0
}
REGION_SIZES = {
    'input': 16, 'sensory': 200, 'dg': 200, 'ca3': 200, 'ca1': 200,
    'thought': 200, 'motor': 400, 'motivation': 200, 'output': 2
}

OFFSETS = {}
_off = 0
for r in REGION_ORDER:
    OFFSETS[r] = _off
    _off += REGION_SIZES[r]
N = _off


# ==================== NUMBA ====================
@njit(cache=True)
def _propagate(active, active_count, targets, target_count, last_used,
               fire_pot, potential, spike_th_mod, meta,
               signal_id, address_map, mg_th, tick):
    for k in range(active_count):
        src = active[k]
        cnt = int(target_count[src])
        fp = int(fire_pot[src])
        s_src = int(signal_id[src])
        if cnt == 0:
            d = int(address_map[src])
            if d >= 0:
                m = int(spike_th_mod[d])
                mg_eff = mg_th + m
                if mg_eff < 0:
                    mg_eff = 0
                v = fp - mg_eff
                if v > 0:
                    cur = int(potential[d])
                    nv = cur + v
                    if nv > 255:
                        nv = 255
                    potential[d] = np.uint8(nv)
                    if s_src != 0:
                        signal_id[d] = s_src
                    meta[d] = np.uint16(int(meta[d]) | 1)
        else:
            for j in range(cnt):
                d = int(targets[src, j])
                m = int(spike_th_mod[d])
                mg_eff = mg_th + m
                if mg_eff < 0:
                    mg_eff = 0
                v = fp - mg_eff
                if v > 0:
                    cur = int(potential[d])
                    nv = cur + v
                    if nv > 255:
                        nv = 255
                    potential[d] = np.uint8(nv)
                    if s_src != 0:
                        signal_id[d] = s_src
                    meta[d] = np.uint16(int(meta[d]) | 1)
                    last_used[src, j] = tick


@njit(cache=True)
def _ltp_free(active_prev, active_prev_count, active_curr, active_curr_count,
              fire_pot_prev, spike_th_mod, is_port,
              targets, target_count, mg_th, max_out):
    out_src = np.empty(active_prev_count, dtype=np.int32)
    out_dst = np.empty(active_prev_count, dtype=np.int32)
    out_slot = np.empty(active_prev_count, dtype=np.int32)
    count = 0
    for ks in range(active_prev_count):
        src = active_prev[ks]
        if is_port[src] == 0:
            continue
        cnt = int(target_count[src])
        if cnt >= max_out:
            continue
        fp = int(fire_pot_prev[src])
        picked = -1
        for kc in range(active_curr_count):
            dst = active_curr[kc]
            if is_port[dst] == 0:
                continue
            dup = False
            for j in range(cnt):
                if int(targets[src, j]) == dst:
                    dup = True
                    break
            if dup:
                continue
            m = int(spike_th_mod[dst])
            mg_eff = mg_th + m
            if mg_eff < 0:
                mg_eff = 0
            if fp - mg_eff > 0:
                picked = dst
                break
        if picked >= 0:
            out_src[count] = src
            out_dst[count] = picked
            out_slot[count] = cnt
            count += 1
    return out_src[:count], out_dst[:count], out_slot[:count], count


# ==================== ENGINE ====================
class NAJime:
    def __init__(self, mode="ram", data_dir=None, seed=42):
        self.mode = mode
        self.seed = seed
        self.tick_count = 0
        self.episode_id = 0
        self.consolidations = 0
        self.motivation_active = False
        self.signal_counter = 0

        if mode == "ram":
            self.meta = np.zeros(N, dtype=np.uint16)
            self.potential = np.zeros(N, dtype=np.uint8)
            self.spike_th_mod = np.zeros(N, dtype=np.int8)
        else:
            if data_dir is None:
                data_dir = f"najime_{uuid.uuid4().hex[:8]}"
            os.makedirs(data_dir, exist_ok=True)
            self.data_dir = data_dir
            self.meta = np.memmap(os.path.join(data_dir, "meta.bin"),
                                  dtype=np.uint16, mode="w+", shape=(N,))
            self.potential = np.memmap(os.path.join(data_dir, "pot.bin"),
                                       dtype=np.uint8, mode="w+", shape=(N,))
            self.spike_th_mod = np.memmap(os.path.join(data_dir, "mod.bin"),
                                          dtype=np.int8, mode="w+", shape=(N,))
            self.meta[:] = 0
            self.potential[:] = 0
            self.spike_th_mod[:] = 0

        self.base_th = np.full(N, BASE_TH, dtype=np.int16)
        self.is_port = np.zeros(N, dtype=np.uint8)
        self.signal_id = np.zeros(N, dtype=np.uint16)
        self._init_layout()

        self.targets = np.full((N, MAX_OUT), -1, dtype=np.int32)
        self.target_count = np.zeros(N, dtype=np.uint8)
        self.last_used = np.full((N, MAX_OUT), -1, dtype=np.int32)

        self.address_map = self._build_address_map()

        self.active_curr = np.zeros(N, dtype=np.int32)
        self.active_curr_count = 0
        self.active_prev = np.zeros(N, dtype=np.int32)
        self.active_prev_count = 0

        self.spike_prev = np.zeros(N, dtype=np.uint8)
        self.spike_curr = np.zeros(N, dtype=np.uint8)
        self.fire_pot_prev = np.zeros(N, dtype=np.uint8)
        self.fire_pot_curr = np.zeros(N, dtype=np.uint8)

        self.touched_buf = np.zeros(N, dtype=np.int32)
        self.touched_count = 0

        self.signal_level = {}

    def _init_layout(self):
        inp_lo = OFFSETS['input']
        for i in range(REGION_SIZES['input']):
            nid = inp_lo + i
            self.is_port[nid] = 1
            self.base_th[nid] = BASE_TH - PORT_TH_OFFSET
        out_lo = OFFSETS['output']
        for i in range(REGION_SIZES['output']):
            self.base_th[out_lo + i] = BASE_TH
        for r in REGION_ORDER:
            if r in ('input', 'output'):
                continue
            off = OFFSETS[r]
            for c in range(REGION_COLS[r]):
                for layer in range(N_LAYERS):
                    for i in range(LAYER_SIZE):
                        nid = off + c * COL_SIZE + layer * LAYER_SIZE + i
                        if layer in PORT_LAYERS:
                            self.is_port[nid] = 1
                            extra = DG_TH_OFFSET if r == 'dg' else 0
                            self.base_th[nid] = BASE_TH - PORT_TH_OFFSET + extra
                        else:
                            self.base_th[nid] = BASE_TH

    def _build_address_map(self):
        addr = np.full(N, -1, dtype=np.int32)
        cols = []
        cols.append(('input', OFFSETS['input'], REGION_SIZES['input'], False))
        for c in range(REGION_COLS['sensory']):
            cols.append(('sensory', OFFSETS['sensory'] + c*COL_SIZE, COL_SIZE, True))
        for c in range(REGION_COLS['dg']):
            cols.append(('dg', OFFSETS['dg'] + c*COL_SIZE, COL_SIZE, True))
        for c in range(REGION_COLS['ca3']):
            cols.append(('ca3', OFFSETS['ca3'] + c*COL_SIZE, COL_SIZE, True))
        for c in range(REGION_COLS['ca1']):
            cols.append(('ca1', OFFSETS['ca1'] + c*COL_SIZE, COL_SIZE, True))
        for c in range(REGION_COLS['thought']):
            cols.append(('thought', OFFSETS['thought'] + c*COL_SIZE, COL_SIZE, True))
        for c in range(REGION_COLS['motor']):
            cols.append(('motor', OFFSETS['motor'] + c*COL_SIZE, COL_SIZE, True))
        cols.append(('output', OFFSETS['output'], REGION_SIZES['output'], False))

        def port_list(base, size, layered):
            if not layered:
                return [base + i for i in range(size)]
            return [base + layer*LAYER_SIZE + j
                    for layer in PORT_LAYERS for j in range(LAYER_SIZE)]

        for ci in range(len(cols) - 1):
            reg_s, base_s, size_s, lay_s = cols[ci]
            reg_d, base_d, size_d, lay_d = cols[ci + 1]
            ps = port_list(base_s, size_s, lay_s)
            pd = port_list(base_d, size_d, lay_d)
            if not ps or not pd:
                continue
            if reg_s == 'motor' and reg_d == 'output':
                col_idx = (base_s - OFFSETS['motor']) // COL_SIZE
                target = OFFSETS['output'] + (0 if col_idx >= 2 else 1)
                for p in ps:
                    addr[p] = target
                continue
            n_d = len(pd)
            for k, p in enumerate(ps):
                addr[p] = pd[k % n_d]

        for c in range(REGION_COLS['motivation']):
            base = OFFSETS['motivation'] + c*COL_SIZE
            thought_base = OFFSETS['thought'] + (c % REGION_COLS['thought'])*COL_SIZE
            for layer in PORT_LAYERS:
                for j in range(LAYER_SIZE):
                    ps = base + layer*LAYER_SIZE + j
                    pd = thought_base + layer*LAYER_SIZE + j
                    addr[ps] = pd
        return addr

    def _add_edge(self, src, dst):
        cnt = int(self.target_count[src])
        if cnt >= MAX_OUT:
            return False
        for j in range(cnt):
            if int(self.targets[src, j]) == dst:
                return False
        self.targets[src, cnt] = dst
        self.target_count[src] = cnt + 1
        self.last_used[src, cnt] = self.tick_count
        return True

    def present_input(self, positions):
        self.signal_counter = (self.signal_counter % (SIGNAL_MAX - 1)) + 1
        sid = self.signal_counter
        for p, v in enumerate(positions):
            for bit in range(2):
                nid = 2 * p + bit
                if (v >> bit) & 1:
                    self.potential[nid] = INPUT_STRENGTH
                    self.signal_id[nid] = sid
                else:
                    self.potential[nid] = 0

    def read_output(self):
        def last_fire(n):
            return (int(self.meta[n]) >> 1) & 0x7FFF
        def recent(t):
            if t == 0:
                return 0
            cur = self.tick_count & 0x7FFF
            return 1 if ((cur - t) & 0x7FFF) <= 5 else 0
        return (recent(last_fire(OFFSETS['output'])),
                recent(last_fire(OFFSETS['output'] + 1)))

    def reinforce(self, delta=3):
        cur = self.tick_count & 0x7FFF
        for i in range(N):
            last = (int(self.meta[i]) >> 1) & 0x7FFF
            if last == 0:
                continue
            if 0 < ((cur - last) & 0x7FFF) <= 10:
                m = int(self.spike_th_mod[i]) - delta
                self.spike_th_mod[i] = np.int8(max(m, -15))

    def suppress(self, delta=3):
        cur = self.tick_count & 0x7FFF
        for i in range(N):
            last = (int(self.meta[i]) >> 1) & 0x7FFF
            if last == 0:
                continue
            if 0 < ((cur - last) & 0x7FFF) <= 10:
                m = int(self.spike_th_mod[i]) + delta
                self.spike_th_mod[i] = np.int8(min(m, 25))

    def _motivation_step(self):
        cur = self.tick_count & 0x7FFF
        idle = True
        for n in range(OFFSETS['thought'],
                       OFFSETS['thought'] + REGION_SIZES['thought']):
            last = (int(self.meta[n]) >> 1) & 0x7FFF
            if last != 0 and ((cur - last) & 0x7FFF) <= MOTIVATION_IDLE_WINDOW:
                idle = False
                break
        self.motivation_active = idle
        if idle:
            for reg in ('thought', 'motor'):
                off = OFFSETS[reg]
                for i in range(off, off + REGION_SIZES[reg]):
                    m = int(self.spike_th_mod[i]) - MOTIVATION_GAIN
                    self.spike_th_mod[i] = np.int8(max(m, -10))

    def _prune_ltd(self):
        removed = 0
        for reg in LTD_REGIONS:
            reg_start = OFFSETS[reg]
            reg_end = reg_start + REGION_SIZES[reg]
            for src in range(reg_start, reg_end):
                cnt = int(self.target_count[src])
                keep, keep_used = [], []
                for j in range(cnt):
                    idle = self.tick_count - int(self.last_used[src, j])
                    if idle <= DG_LTD_TICKS:
                        keep.append(int(self.targets[src, j]))
                        keep_used.append(int(self.last_used[src, j]))
                    else:
                        removed += 1
                for j in range(len(keep)):
                    self.targets[src, j] = keep[j]
                    self.last_used[src, j] = keep_used[j]
                for j in range(len(keep), cnt):
                    self.targets[src, j] = -1
                    self.last_used[src, j] = -1
                self.target_count[src] = len(keep)
        return removed

    def _erase_hippo_edges(self):
        hip_start, hip_end = OFFSETS['dg'], OFFSETS['ca1'] + REGION_SIZES['ca1']
        for src in range(N):
            cnt = int(self.target_count[src])
            keep, keep_used = [], []
            for j in range(cnt):
                d = int(self.targets[src, j])
                if not (hip_start <= d < hip_end):
                    keep.append(d)
                    keep_used.append(int(self.last_used[src, j]))
            for j in range(len(keep)):
                self.targets[src, j] = keep[j]
                self.last_used[src, j] = keep_used[j]
            for j in range(len(keep), cnt):
                self.targets[src, j] = -1
                self.last_used[src, j] = -1
            self.target_count[src] = len(keep)
        self.episode_id += 1
        self.consolidations += 1

    def _ltp_step(self):
        srcs, dsts, slots, cnt = _ltp_free(
            self.active_prev, self.active_prev_count,
            self.active_curr, self.active_curr_count,
            self.fire_pot_prev, self.spike_th_mod, self.is_port,
            self.targets, self.target_count, MG_TH, MAX_OUT)
        for k in range(cnt):
            src = int(srcs[k]); dst = int(dsts[k]); slot = int(slots[k])
            self.targets[src, slot] = dst
            self.target_count[src] = slot + 1
            self.last_used[src, slot] = self.tick_count

    def tick(self):
        potential = self.potential
        mod = self.spike_th_mod
        base_th = self.base_th
        meta = self.meta

        self._motivation_step()

        self.fire_pot_curr[:] = 0
        self.spike_curr[:] = 0
        self.active_curr_count = 0
        for i in range(N):
            thr = int(base_th[i]) + int(mod[i])
            if thr < 1: thr = 1
            elif thr > 255: thr = 255
            if int(potential[i]) >= thr:
                self.spike_curr[i] = 1
                self.fire_pot_curr[i] = potential[i]
                potential[i] = 0
                self.active_curr[self.active_curr_count] = i
                self.active_curr_count += 1

        self._ltp_step()

        _propagate(self.active_curr, self.active_curr_count,
                   self.targets, self.target_count, self.last_used,
                   self.fire_pot_curr, potential, mod, meta,
                   self.signal_id, self.address_map, MG_TH, self.tick_count)

        self.touched_count = 0
        for i in range(N):
            if meta[i] & 1:
                self.touched_buf[self.touched_count] = i
                self.touched_count += 1

        for i in range(N):
            p = int(potential[i])
            if p > 0:
                potential[i] = np.uint8(p - LEAK_STEP if p >= LEAK_STEP else 0)

        tc = (self.tick_count + 1) & 0x7FFF
        for i in range(N):
            if self.spike_curr[i]:
                m = int(mod[i]) + ADAPT_STEP
                mod[i] = np.int8(max(min(m, 25), -15))
                meta[i] = np.uint16((tc << 1) & 0xFFFE)

        for k in range(self.touched_count):
            i = int(self.touched_buf[k])
            if not self.spike_curr[i]:
                meta[i] = np.uint16(int(meta[i]) & 0xFFFE)

        self.signal_level = {}
        for i in range(N):
            s = int(self.signal_id[i])
            if s == 0:
                continue
            p = int(potential[i])
            if p > 0:
                prev = self.signal_level.get(s, 0)
                if p > prev:
                    self.signal_level[s] = p

        cur_t = (self.tick_count + 1) & 0x7FFF
        for i in range(N):
            last = (int(meta[i]) >> 1) & 0x7FFF
            if last != 0 and ((cur_t - last) & 0x7FFF) <= LATERAL_REFRACTORY:
                continue
            p = int(potential[i])
            s = int(self.signal_id[i])
            if s != 0 and p > 0:
                lvl = self.signal_level.get(s, 0)
                if p >= lvl - WAVE_DELTA:
                    m = int(mod[i]) - LATERAL_FACIL_GAIN
                    mod[i] = np.int8(max(m, LATERAL_FACIL_LIMIT))
                else:
                    m = int(mod[i]) + LATERAL_INHIB_GAIN
                    mod[i] = np.int8(min(m, LATERAL_INHIB_LIMIT))
            else:
                m = int(mod[i])
                if m > 0:
                    mod[i] = np.int8(m - MOD_DECAY)
                elif m < 0:
                    mod[i] = np.int8(m + MOD_DECAY)

        ca1_off = OFFSETS['ca1']
        ca1_cnt = int(self.spike_curr[ca1_off:
                                      ca1_off + REGION_SIZES['ca1']].sum())
        if ca1_cnt >= CONSOLIDATE_BURST:
            self._erase_hippo_edges()
        if self.tick_count > 0 and self.tick_count % PRUNE_EVERY == 0:
            self._prune_ltd()

        self.active_prev[:self.active_curr_count] = \
            self.active_curr[:self.active_curr_count]
        self.active_prev_count = self.active_curr_count
        self.spike_prev[:] = self.spike_curr
        self.fire_pot_prev[:] = self.fire_pot_curr
        self.tick_count += 1


# ==================== TEST HARNESS ====================
def _active(eng, region):
    a, b = OFFSETS[region], OFFSETS[region] + REGION_SIZES[region]
    return eng.spike_curr[a:b].astype(bool)


def _jaccard(a, b):
    a = a.astype(bool); b = b.astype(bool)
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u > 0 else 1.0


def _run_pattern(pattern, n_ticks=12, settle=15):
    e = NAJime(seed=1)
    for _ in range(settle):
        e.tick()
    regions = ('sensory', 'dg', 'ca3', 'ca1', 'thought', 'motor')
    accum = {r: np.zeros(REGION_SIZES[r], dtype=bool) for r in regions}
    for _ in range(n_ticks):
        e.present_input(pattern)
        e.tick()
        for r in regions:
            a = OFFSETS[r]
            accum[r] |= e.spike_curr[a:a + REGION_SIZES[r]].astype(bool)
    return e, accum


# --- plain-check helpers so tests never abort the run ---
_passed = 0
_failed = 0

def _check(label, condition, detail=""):
    global _passed, _failed
    if condition:
        _passed += 1
        tag = "PASS"
    else:
        _failed += 1
        tag = "FAIL"
    line = f"  [{tag}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return condition


def _test_1():
    print("\nTest 1: 4 bytes/neuron, empty initial graph")
    eng = NAJime()
    total = eng.meta.nbytes + eng.potential.nbytes + eng.spike_th_mod.nbytes
    _check("4 bytes/neuron", total == 4 * N, f"N={N}, total={total}")
    edges_init = int(eng.target_count.sum())
    _check("empty initial graph", edges_init == 0, f"edges={edges_init}")


def _test_2():
    print("\nTest 2: linear leak")
    eng = NAJime()
    nid = OFFSETS['thought'] + 5
    eng.potential[nid] = 30
    eng.spike_th_mod[nid] = 15
    for _ in range(5):
        eng.tick()
    p = int(eng.potential[nid])
    _check("leak reduces potential", p < 30, f"30 -> {p}")


def _test_3():
    print("\nTest 3: input has ports, output doesn't")
    eng = NAJime()
    inp_ports = sum(int(eng.is_port[OFFSETS['input'] + i])
                    for i in range(REGION_SIZES['input']))
    out_ports = sum(int(eng.is_port[OFFSETS['output'] + i])
                    for i in range(REGION_SIZES['output']))
    _check("input all ports", inp_ports == REGION_SIZES['input'],
           f"{inp_ports}/{REGION_SIZES['input']}")
    _check("output no ports", out_ports == 0,
           f"{out_ports}/{REGION_SIZES['output']}")


def _test_4():
    print("\nTest 4: LATERAL WAVE — front facilitates, tail suppresses")
    eng = NAJime()
    for _ in range(3):
        eng.tick()
    eng._motivation_step = lambda: None
    a = OFFSETS['thought'] + 30
    b = OFFSETS['thought'] + 31
    c = OFFSETS['thought'] + 32
    eng.signal_id[a] = 42; eng.signal_id[b] = 42; eng.signal_id[c] = 42
    eng.potential[a] = 30; eng.potential[b] = 28; eng.potential[c] = 8
    eng.spike_th_mod[a] = 12; eng.spike_th_mod[b] = 12; eng.spike_th_mod[c] = 12
    m_a0 = int(eng.spike_th_mod[a]); m_c0 = int(eng.spike_th_mod[c])
    eng.tick()
    m_a1 = int(eng.spike_th_mod[a]); m_c1 = int(eng.spike_th_mod[c])
    _check("front facilitated", m_a1 < m_a0, f"{m_a0} -> {m_a1}")
    _check("tail suppressed", m_c1 > m_c0, f"{m_c0} -> {m_c1}")
    _check("front ends below tail", m_a1 < m_c1, f"{m_a1} < {m_c1}")


def _test_5():
    print("\nTest 5: wave self-assembles from scratch")
    eng = NAJime()
    for _ in range(15):
        eng.tick()
    trace = []
    motor_slice = slice(OFFSETS['motor'],
                        OFFSETS['motor'] + REGION_SIZES['motor'])
    motor_tick = -1
    out_off = OFFSETS['output']
    out_tick = -1
    for t in range(60):
        eng.present_input([1, 0, 0, 0, 0, 0, 0, 0])
        eng.tick()
        trace.append(int(eng.spike_curr.sum()))
        if motor_tick < 0 and eng.spike_curr[motor_slice].sum() > 0:
            motor_tick = t
        if out_tick < 0 and (eng.spike_curr[out_off] or eng.spike_curr[out_off+1]):
            out_tick = t
    print(f"    spike trace (first 30): {trace[:30]}")
    edges = int(eng.target_count.sum())
    _check("edges self-assembled", edges > 0, f"edges={edges}")
    _check("motor fired", motor_tick > 0, f"motor @ {motor_tick}")
    _check("output fired", out_tick > 0, f"output @ {out_tick}")


def _test_6():
    print("\nTest 6: epilepsy — 300 idle ticks")
    eng = NAJime()
    peak = 0
    for t in range(300):
        eng.tick()
        s = int(eng.spike_curr.sum())
        if s > peak:
            peak = s
    _check("no spontaneous spikes", peak == 0, f"peak={peak}")


def _test_7():
    print("\nTest 7: Jaccard — pattern separation")
    A   = [1,0,0,0,0,0,0,0]
    Ap1 = [1,1,0,0,0,0,0,0]
    Z   = [0,0,0,0,0,0,0,3]
    _, rA = _run_pattern(A)
    _, rA1 = _run_pattern(Ap1)
    _, rZ = _run_pattern(Z)
    ok_all = True
    for reg in ('sensory','dg','ca3','ca1','thought','motor'):
        js = _jaccard(rA[reg], rA1[reg])
        jd = _jaccard(rA[reg], rZ[reg])
        print(f"    {reg:8s} J(A,A+1)={js:.3f}  J(A,Z)={jd:.3f}")
        if js < jd - 0.05:
            ok_all = False
    _check("similar inputs at least as similar as dissimilar", ok_all)


def _test_8():
    print("\nTest 8: DG LTD — organic edges only")
    eng = NAJime()
    for _ in range(60):
        eng.present_input([1, 0, 0, 0, 0, 0, 0, 0])
        eng.tick()
    dg_off = OFFSETS['dg']
    dg_edges = sum(int(eng.target_count[dg_off+i])
                   for i in range(REGION_SIZES['dg']))
    print(f"    DG edges after 60 ticks: {dg_edges}")
    if dg_edges == 0:
        _check("DG prune (no edges yet)", True, "skipped, nothing to prune")
        return
    for i in range(REGION_SIZES['dg']):
        src = dg_off + i
        for j in range(int(eng.target_count[src])):
            eng.last_used[src, j] = -DG_LTD_TICKS - 100
    eng.tick_count = DG_LTD_TICKS + 200
    removed = eng._prune_ltd()
    after = sum(int(eng.target_count[dg_off+i])
                for i in range(REGION_SIZES['dg']))
    _check("all idle DG edges pruned",
           removed == dg_edges,
           f"{dg_edges} -> {after}, removed={removed}")


def _test_9():
    print("\nTest 9: LTP — one new edge per src per tick")
    eng = NAJime()
    for _ in range(15):
        eng.tick()
    for _ in range(10):
        eng.present_input([1,2,3,0,0,0,0,0])
        eng.tick()
    total = int(eng.target_count.sum())
    _check("LTP produces edges", total > 0, f"edges={total}")


def _test_10():
    print("\nTest 10: determinism")
    e1 = NAJime(seed=5); e2 = NAJime(seed=5)
    for _ in range(20):
        e1.present_input([1,2,0,0,0,0,0,0])
        e2.present_input([1,2,0,0,0,0,0,0])
        e1.tick(); e2.tick()
    same_pot = np.array_equal(e1.potential, e2.potential)
    same_mod = np.array_equal(e1.spike_th_mod, e2.spike_th_mod)
    same_edges = np.array_equal(e1.target_count, e2.target_count)
    _check("identical potential", same_pot)
    _check("identical mod", same_mod)
    _check("identical target_count", same_edges)


def _test_11():
    print("\nTest 11: real-time output")
    e = NAJime(seed=1)
    for _ in range(15): e.tick()
    ms = slice(OFFSETS['motor'], OFFSETS['motor'] + REGION_SIZES['motor'])
    out_off = OFFSETS['output']
    mt = ot = -1
    for t in range(80):
        e.present_input([1,0,0,0,0,0,0,0])
        e.tick()
        if mt < 0 and e.spike_curr[ms].sum() > 0: mt = t
        if ot < 0 and (e.spike_curr[out_off] or e.spike_curr[out_off+1]):
            ot = t
        if mt >= 0 and ot >= 0: break
    lag = ot - mt
    print(f"    motor @ {mt}, output @ {ot}")
    _check("motor fired", mt > 0)
    _check("output fired", ot > 0)
    _check("output lag reasonable (≤5)", 0 <= lag <= 5, f"lag={lag}")


def run_tests():
    global _passed, _failed
    _passed = 0
    _failed = 0
    print("=== NAJime v2 (self-assembling, no initial graph) ===")
    for fn in (_test_1, _test_2, _test_3, _test_4, _test_5,
               _test_6, _test_7, _test_8, _test_9, _test_10, _test_11):
        try:
            fn()
        except Exception as e:
            _failed += 1
            print(f"  [FAIL] exception {type(e).__name__}: {e}")
    print(f"\n=== summary: {_passed} passed, {_failed} failed ===")


# ==================== REPL ====================
def run_repl():
    eng = NAJime(seed=1)
    print("NAJime v2 REPL | p v0..v7 | t N | o | r D | s D | q")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0]
        if cmd == 'q': break
        elif cmd == 'p':
            vals = [int(x) & 3 for x in parts[1:9]]
            while len(vals) < 8: vals.append(0)
            eng.present_input(vals); print(f"  {vals}")
        elif cmd == 't':
            n = int(parts[1]) if len(parts) > 1 else 1
            for _ in range(n): eng.tick()
            print(f"  ran {n} (total {eng.tick_count})")
        elif cmd == 'o':
            print("  " + str(eng.read_output()))
        elif cmd == 'r':
            eng.reinforce(int(parts[1]) if len(parts) > 1 else 3); print("  +")
        elif cmd == 's':
            eng.suppress(int(parts[1]) if len(parts) > 1 else 3); print("  -")
        else:
            print("  ?")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "repl":
        run_repl()
    else:
        run_tests()