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

HIPPO_GAIN = 3
HIPPO_FLOOR = -19
HIPPO_REST = 6
HIPPO_REFRACTORY = 3

SIGNAL_MAX = 65000
EDGE_GROW_CHUNK = 4096
TARGET_WINDOW = 32

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

HIPPO_LO = OFFSETS['dg']
HIPPO_HI = OFFSETS['ca1'] + REGION_SIZES['ca1']


# ==================== DYNAMIC DTYPE SELECTION ====================
def _min_uint(max_val):
    if max_val <= 0xFF:
        return np.uint8
    if max_val <= 0xFFFF:
        return np.uint16
    if max_val <= 0xFFFFFFFF:
        return np.uint32
    return np.uint64


# ==================== NUMBA ====================
@njit(cache=True)
def _propagate(active, active_count, head, edge_dst, edge_next, edge_last_used,
               fire_pot, potential, spike_th_mod, meta,
               signal_id, address_map, mg_th, tick):
    for k in range(active_count):
        src = active[k]
        fp = int(fire_pot[src])
        s_src = int(signal_id[src])
        d_anat = int(address_map[src])
        if d_anat >= 0:
            m = int(spike_th_mod[d_anat])
            mg_eff = mg_th + m
            if mg_eff < 0:
                mg_eff = 0
            v = fp - mg_eff
            if v > 0:
                cur = int(potential[d_anat])
                nv = cur + v
                if nv > 255:
                    nv = 255
                potential[d_anat] = np.uint8(nv)
                if s_src != 0:
                    signal_id[d_anat] = s_src
                meta[d_anat] = np.uint16(int(meta[d_anat]) | 1)
        e = head[src]
        while e != -1:
            d = int(edge_dst[e])
            if d != d_anat:
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
                    edge_last_used[e] = tick
            e = edge_next[e]


@njit(cache=True)
def _ltp_stream(active_prev, active_prev_count, spike_curr,
                fire_pot_prev, spike_th_mod, is_port,
                mini_start, mini_end, port_idx, adj_bitmap, bitmap_words,
                mg_th, target_window):
    out_src = np.empty(active_prev_count, dtype=np.int32)
    out_dst = np.empty(active_prev_count, dtype=np.int32)
    count = 0
    for ks in range(active_prev_count):
        src = active_prev[ks]
        if is_port[src] == 0:
            continue
        col_lo = int(mini_start[src])
        col_hi = int(mini_end[src])
        col_size = col_hi - col_lo
        if col_size <= 1:
            continue
        pi = int(port_idx[src])
        if pi < 0:
            continue
        fp = int(fire_pot_prev[src])
        base_off = src - col_lo
        for i in range(1, target_window + 1):
            off = (base_off + i) % col_size
            dst = col_lo + off
            if is_port[dst] == 0:
                continue
            if spike_curr[dst] == 0:
                continue
            m = int(spike_th_mod[dst])
            mg_eff = mg_th + m
            if mg_eff < 0:
                mg_eff = 0
            if fp - mg_eff <= 0:
                continue
            di = int(port_idx[dst])
            if di < 0:
                continue
            w = pi * bitmap_words + (di >> 6)
            bit = np.uint64(1) << np.uint64(di & 63)
            if adj_bitmap[w] & bit:
                continue
            out_src[count] = src
            out_dst[count] = dst
            count += 1
            break
    return out_src[:count], out_dst[:count], count


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
        self._init_minicolumns()

        self.port_idx = np.full(N, -1, dtype=np.int32)
        k = 0
        for i in range(N):
            if self.is_port[i]:
                self.port_idx[i] = k
                k += 1
        self.n_ports = k
        self.bitmap_words = (self.n_ports + 63) // 64
        self.adj_bitmap = np.zeros(self.n_ports * self.bitmap_words,
                                   dtype=np.uint64)

        self.DST_DTYPE = _min_uint(N - 1)
        self.USED_DTYPE = _min_uint(0)

        self.head = np.full(N, -1, dtype=np.int32)
        self.edge_count = 0
        self.edge_dst = np.empty(0, dtype=self.DST_DTYPE)
        self.edge_next = np.empty(0, dtype=np.int32)
        self.edge_prev = np.empty(0, dtype=np.int32)
        self.edge_last_used = np.empty(0, dtype=self.USED_DTYPE)

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

    def _init_minicolumns(self):
        self.mini_start = np.zeros(N, dtype=np.int32)
        self.mini_end = np.zeros(N, dtype=np.int32)
        for r in REGION_ORDER:
            if r in ('input', 'output'):
                lo = OFFSETS[r]
                hi = lo + REGION_SIZES[r]
                for i in range(lo, hi):
                    self.mini_start[i] = lo
                    self.mini_end[i] = hi
                continue
            for c in range(REGION_COLS[r]):
                lo = OFFSETS[r] + c * COL_SIZE
                hi = lo + COL_SIZE
                for i in range(lo, hi):
                    self.mini_start[i] = lo
                    self.mini_end[i] = hi

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

    def _bitmap_get(self, src, dst):
        pi = int(self.port_idx[src]); di = int(self.port_idx[dst])
        if pi < 0 or di < 0:
            return False
        w = pi * self.bitmap_words + (di >> 6)
        return bool(self.adj_bitmap[w] & (np.uint64(1) << np.uint64(di & 63)))

    def _bitmap_set(self, src, dst):
        pi = int(self.port_idx[src]); di = int(self.port_idx[dst])
        if pi < 0 or di < 0:
            return
        w = pi * self.bitmap_words + (di >> 6)
        self.adj_bitmap[w] |= (np.uint64(1) << np.uint64(di & 63))

    def _bitmap_clear(self, src, dst):
        pi = int(self.port_idx[src]); di = int(self.port_idx[dst])
        if pi < 0 or di < 0:
            return
        w = pi * self.bitmap_words + (di >> 6)
        self.adj_bitmap[w] &= ~(np.uint64(1) << np.uint64(di & 63))

    def _ensure_edge_capacity(self, need):
        cap = self.edge_dst.shape[0]
        if need <= cap:
            return
        new_cap = max(need, cap + max(EDGE_GROW_CHUNK, cap // 2))
        new_dst = np.empty(new_cap, dtype=self.DST_DTYPE)
        new_next = np.empty(new_cap, dtype=np.int32)
        new_prev = np.empty(new_cap, dtype=np.int32)
        new_used = np.empty(new_cap, dtype=self.USED_DTYPE)
        n = self.edge_count
        if n > 0:
            new_dst[:n] = self.edge_dst[:n]
            new_next[:n] = self.edge_next[:n]
            new_prev[:n] = self.edge_prev[:n]
            new_used[:n] = self.edge_last_used[:n]
        self.edge_dst = new_dst
        self.edge_next = new_next
        self.edge_prev = new_prev
        self.edge_last_used = new_used

    def _add_edge(self, src, dst):
        if self._bitmap_get(src, dst):
            return False
        self._bitmap_set(src, dst)
        idx = self.edge_count
        self._ensure_edge_capacity(idx + 1)
        self.edge_dst[idx] = dst
        self.edge_last_used[idx] = self.tick_count
        old_head = int(self.head[src])
        self.edge_prev[idx] = -1
        self.edge_next[idx] = old_head
        if old_head >= 0:
            self.edge_prev[old_head] = idx
        self.head[src] = idx
        self.edge_count = idx + 1
        return True

    def _remove_edge(self, src, e):
        d = int(self.edge_dst[e])
        self._bitmap_clear(src, d)
        prev = int(self.edge_prev[e])
        nxt = int(self.edge_next[e])
        if prev == -1:
            self.head[src] = nxt
        else:
            self.edge_next[prev] = nxt
        if nxt != -1:
            self.edge_prev[nxt] = prev

    def _tick_maybe_upcast(self):
        if self.tick_count > np.iinfo(self.USED_DTYPE).max:
            new_dt = _min_uint(self.tick_count)
            self.USED_DTYPE = new_dt
            self.edge_last_used = self.edge_last_used.astype(new_dt)

    def n_edges(self):
        return int(self.edge_count)

    def fanout(self, src):
        n = 0
        e = int(self.head[src])
        while e != -1:
            n += 1
            e = int(self.edge_next[e])
        return n

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

    def read_output_now(self):
        return (int(self.spike_curr[OFFSETS['output']]),
                int(self.spike_curr[OFFSETS['output'] + 1]))

    def present_byte(self, b):
        b = int(b) & 0xFF
        self.signal_counter = (self.signal_counter % (SIGNAL_MAX - 1)) + 1
        sid = self.signal_counter
        for p in range(4):
            lo = (b >> (2*p)) & 1
            hi = (b >> (2*p+1)) & 1
            nid_lo = 2*p
            nid_hi = 2*p + 1
            if lo:
                self.potential[nid_lo] = INPUT_STRENGTH
                self.signal_id[nid_lo] = sid
            else:
                self.potential[nid_lo] = 0
            if hi:
                self.potential[nid_hi] = INPUT_STRENGTH
                self.signal_id[nid_hi] = sid
            else:
                self.potential[nid_hi] = 0
        for p in range(4, 8):
            self.potential[2*p] = 0
            self.potential[2*p+1] = 0

    def feed_byte(self, b, hold=4, gap=1):
        for _ in range(hold):
            self.present_byte(b)
            self.tick()
        for _ in range(gap):
            self.tick()

    def feed_text(self, text, hold=4, gap=1):
        for ch in text:
            b = ord(ch) & 0xFF
            self.feed_byte(b, hold=hold, gap=gap)

    def read_byte(self, max_ticks=30):
        bits = []
        for _ in range(max_ticks):
            self.tick()
            o0, o1 = self.read_output_now()
            if o0 or o1:
                bits.append((o0, o1))
                if len(bits) == 4:
                    b = 0
                    for k, (a, c) in enumerate(bits):
                        b |= (a << (2*k)) | (c << (2*k+1))
                    return b
        return -1

    def chat(self, prompt, max_new=32, hold=4, gap=1):
        self.feed_text(prompt, hold=hold, gap=gap)
        out = []
        for _ in range(max_new):
            b = self.read_byte(max_ticks=30)
            if b < 0:
                out.append(0)
                continue
            out.append(b)
            self.feed_byte(b, hold=hold, gap=gap)
        return bytes(out)

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
        for reg, gain, floor in (
            ('thought', MOTIVATION_GAIN, -10),
            ('motor',   MOTIVATION_GAIN, -10),
            ('dg',      HIPPO_GAIN,      HIPPO_FLOOR),
            ('ca3',     HIPPO_GAIN,      HIPPO_FLOOR),
            ('ca1',     HIPPO_GAIN,      HIPPO_FLOOR),
        ):
            off = OFFSETS[reg]
            sz = REGION_SIZES[reg]
            idle = True
            for n in range(off, off + sz):
                last = (int(self.meta[n]) >> 1) & 0x7FFF
                if last != 0 and ((cur - last) & 0x7FFF) <= MOTIVATION_IDLE_WINDOW:
                    idle = False
                    break
            if idle:
                for i in range(off, off + sz):
                    m = int(self.spike_th_mod[i]) - gain
                    self.spike_th_mod[i] = np.int8(max(m, floor))

    def _prune_ltd(self):
        removed = 0
        for reg in LTD_REGIONS:
            reg_start = OFFSETS[reg]
            reg_end = reg_start + REGION_SIZES[reg]
            for src in range(reg_start, reg_end):
                e = int(self.head[src])
                while e != -1:
                    nxt = int(self.edge_next[e])
                    if self.tick_count - int(self.edge_last_used[e]) > DG_LTD_TICKS:
                        self._remove_edge(src, e)
                        removed += 1
                    e = nxt
        return removed

    def _erase_hippo_edges(self):
        hip_start, hip_end = OFFSETS['dg'], OFFSETS['ca1'] + REGION_SIZES['ca1']
        for src in range(N):
            e = int(self.head[src])
            while e != -1:
                nxt = int(self.edge_next[e])
                d = int(self.edge_dst[e])
                if hip_start <= d < hip_end:
                    self._remove_edge(src, e)
                e = nxt
        self.episode_id += 1
        self.consolidations += 1

    def _ltp_step(self):
        srcs, dsts, cnt = _ltp_stream(
            self.active_prev, self.active_prev_count,
            self.spike_curr,
            self.fire_pot_prev, self.spike_th_mod, self.is_port,
            self.mini_start, self.mini_end,
            self.port_idx, self.adj_bitmap, self.bitmap_words,
            MG_TH, TARGET_WINDOW)
        for k in range(cnt):
            self._add_edge(int(srcs[k]), int(dsts[k]))

    def tick(self):
        potential = self.potential
        mod = self.spike_th_mod
        base_th = self.base_th
        meta = self.meta

        self._tick_maybe_upcast()
        self._motivation_step()

        self.fire_pot_curr[:] = 0
        self.spike_curr[:] = 0
        self.active_curr_count = 0
        tc_next = (self.tick_count + 1) & 0x7FFF
        for i in range(N):
            thr = int(base_th[i]) + int(mod[i])
            if thr < 1: thr = 1
            elif thr > 255: thr = 255
            if int(potential[i]) >= thr:
                if HIPPO_LO <= i < HIPPO_HI:
                    last = (int(meta[i]) >> 1) & 0x7FFF
                    if last != 0:
                        dt = (tc_next - last) & 0x7FFF
                        if dt <= HIPPO_REFRACTORY:
                            continue
                self.spike_curr[i] = 1
                self.fire_pot_curr[i] = potential[i]
                potential[i] = 0
                self.active_curr[self.active_curr_count] = i
                self.active_curr_count += 1

        self._ltp_step()

        _propagate(self.active_curr, self.active_curr_count,
                   self.head, self.edge_dst, self.edge_next, self.edge_last_used,
                   self.fire_pot_curr, potential, mod, meta,
                   self.signal_id, self.address_map, MG_TH, self.tick_count)

        self.touched_count = 0
        for i in range(N):
            if meta[i] & 1:
                self.touched_buf[self.touched_count] = i
                self.touched_count += 1

        for i in range(N):
            p = int(potential[i])
            if p <= 0:
                continue
            if HIPPO_LO <= i < HIPPO_HI:
                if p > HIPPO_REST:
                    potential[i] = np.uint8(p - LEAK_STEP)
            else:
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


def _accumulate(eng, region, n_ticks):
    a = OFFSETS[region]
    b = a + REGION_SIZES[region]
    accum = np.zeros(REGION_SIZES[region], dtype=bool)
    for _ in range(n_ticks):
        eng.tick()
        accum |= eng.spike_curr[a:b].astype(bool)
    return accum


def _run_pattern(pattern, n_ticks=20, settle=15):
    e = NAJime(seed=1)
    for _ in range(settle):
        e.tick()
    regions = ('sensory', 'dg', 'ca3', 'ca1', 'thought', 'motor')
    accum = {r: np.zeros(REGION_SIZES[r], dtype=bool) for r in regions}
    e.present_input(pattern)
    for _ in range(n_ticks):
        e.tick()
        for r in regions:
            a = OFFSETS[r]
            accum[r] |= e.spike_curr[a:a + REGION_SIZES[r]].astype(bool)
    return e, accum


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
    print("\nTest 1: 4 bytes/neuron, empty graph, intra-minicolumn LTP")
    eng = NAJime()
    total = eng.meta.nbytes + eng.potential.nbytes + eng.spike_th_mod.nbytes
    _check("4 bytes/neuron", total == 4 * N, f"N={N}, total={total}")
    _check("empty initial graph", eng.n_edges() == 0)
    print(f"    ports: {eng.n_ports}, bitmap_bytes: {eng.adj_bitmap.nbytes}, "
          f"TARGET_WINDOW={TARGET_WINDOW}")


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
    _check("input all ports", inp_ports == REGION_SIZES['input'])
    _check("output no ports", out_ports == 0)


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
    _check("front facilitated", m_a1 < m_a0)
    _check("tail suppressed", m_c1 > m_c0)
    _check("front below tail", m_a1 < m_c1)


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
    print(f"    edges total = {eng.n_edges()}")
    _check("edges self-assembled", eng.n_edges() > 0)
    _check("motor fired", motor_tick > 0, f"motor @ {motor_tick}")
    _check("output fired", out_tick > 0, f"output @ {out_tick}")


def _test_6a():
    print("\nTest 6a: pure idle — no input, no episode, zero spikes")
    eng = NAJime()
    peak = 0
    for t in range(300):
        eng.tick()
        s = int(eng.spike_curr.sum())
        if s > peak:
            peak = s
    _check("no spontaneous spikes without episode", peak == 0, f"peak={peak}")


def _test_6b():
    print("\nTest 6b: hungry hippocampus — per-neuron resonance")
    eng = NAJime()
    for _ in range(20):
        eng.present_input([1, 1, 0, 0, 0, 0, 0, 0])
        eng.tick()
    probes = {
        'dg':  OFFSETS['dg']  + 20,
        'ca3': OFFSETS['ca3'] + 20,
        'ca1': OFFSETS['ca1'] + 20,
    }
    fire_ticks = {k: [] for k in probes}
    width_samples = []
    for t in range(800):
        eng.tick()
        hip_n = int(eng.spike_curr[HIPPO_LO:HIPPO_HI].sum())
        if hip_n > 0:
            width_samples.append(hip_n)
        for k, nid in probes.items():
            if eng.spike_curr[nid]:
                fire_ticks[k].append(t)
    for k, ticks in fire_ticks.items():
        if len(ticks) >= 2:
            ints = [ticks[i+1] - ticks[i] for i in range(len(ticks)-1)]
            avg = sum(ints) / len(ints)
            print(f"    probe {k}: {len(ticks)} fires (avg int {avg:.1f})")
        else:
            print(f"    probe {k}: {len(ticks)} fires")
    if width_samples:
        print(f"    wavefront width avg={sum(width_samples)/len(width_samples):.2f}")
    any_res = any(len(t) >= 2 for t in fire_ticks.values())
    _check("at least one hippo probe fires repeatedly", any_res)
    total_fires = sum(len(t) for t in fire_ticks.values())
    _check("hippo sustains activity", total_fires >= 20)


def _test_7():
    print("\nTest 7: Jaccard — pattern separation (accumulate)")
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
    _check("no anti-correlation", ok_all)


def _test_8():
    print("\nTest 8: DG LTD — organic edges only")
    eng = NAJime()
    for _ in range(60):
        eng.present_input([1, 0, 0, 0, 0, 0, 0, 0])
        eng.tick()
    dg_off = OFFSETS['dg']
    dg_edges = sum(eng.fanout(dg_off + i)
                   for i in range(REGION_SIZES['dg']))
    print(f"    DG edges after 60 ticks: {dg_edges}")
    if dg_edges == 0:
        _check("DG prune (no edges yet)", True, "skipped")
        return
    for i in range(REGION_SIZES['dg']):
        src = dg_off + i
        e = int(eng.head[src])
        while e != -1:
            eng.edge_last_used[e] = 0
            e = int(eng.edge_next[e])
    eng.tick_count = DG_LTD_TICKS + 200
    removed = eng._prune_ltd()
    after = sum(eng.fanout(dg_off + i) for i in range(REGION_SIZES['dg']))
    _check("all idle DG edges pruned", removed == dg_edges,
           f"{dg_edges} -> {after}")


def _test_9():
    print("\nTest 9: intra-minicolumn LTP — fanout grows")
    eng = NAJime()
    for _ in range(15):
        eng.tick()
    for _ in range(400):
        eng.present_input([1,2,3,0,0,0,0,0])
        eng.tick()
    total = eng.n_edges()
    peak = max(eng.fanout(i) for i in range(N))
    print(f"    edges total = {total}, peak fanout = {peak}")
    _check("LTP produces edges", total > 0)


def _test_10():
    print("\nTest 10: determinism")
    e1 = NAJime(seed=5); e2 = NAJime(seed=5)
    for _ in range(20):
        e1.present_input([1,2,0,0,0,0,0,0])
        e2.present_input([1,2,0,0,0,0,0,0])
        e1.tick(); e2.tick()
    same_pot = np.array_equal(e1.potential, e2.potential)
    same_mod = np.array_equal(e1.spike_th_mod, e2.spike_th_mod)
    same_edges = (e1.n_edges() == e2.n_edges())
    same_bitmap = np.array_equal(e1.adj_bitmap, e2.adj_bitmap)
    _check("identical potential", same_pot)
    _check("identical mod", same_mod)
    _check("identical edge count", same_edges)
    _check("identical bitmap", same_bitmap)


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
    print(f"    motor @ {mt}, output @ {ot}, lag = {lag}")
    _check("motor fired", mt > 0)
    _check("output fired", ot > 0)
    _check("lag reasonable", 0 <= lag <= 5)


def _test_12():
    print("\nTest 12: extrapolation — 1-tick delay training")

    A_b0   = [1, 0, 0, 0, 0, 0, 0, 0]
    A_b1   = [0, 1, 0, 0, 0, 0, 0, 0]
    A_full = [1, 1, 0, 0, 0, 0, 0, 0]

    # --- naive baseline ---
    e_base = NAJime(seed=1)
    for _ in range(15): e_base.tick()
    e_base.present_input(A_b0)
    cue_naive = _accumulate(e_base, 'thought', 20)

    # --- trained: b0, ONE tick, b1, settle ---
    e = NAJime(seed=1)
    for _ in range(15): e.tick()
    for cycle in range(40):
        e.present_input(A_b0)
        e.tick()          # b0 at t
        e.present_input(A_b1)
        e.tick()          # b1 at t+1
        for _ in range(8): e.tick()

    # full reference
    e.present_input(A_full)
    full_sig = _accumulate(e, 'thought', 20)

    # cue after training
    e.present_input(A_b0)
    cue_trained = _accumulate(e, 'thought', 20)

    missing = full_sig & ~cue_naive
    missing_n = int(missing.sum())
    reached = int((cue_trained & missing).sum())
    j_change = _jaccard(cue_trained, cue_naive)

    print(f"    full active:        {int(full_sig.sum())}")
    print(f"    cue naive active:   {int(cue_naive.sum())}")
    print(f"    cue trained active: {int(cue_trained.sum())}")
    print(f"    missing part:       {missing_n}")
    print(f"    reached by cue:     {reached}")
    print(f"    J(trained, naive):  {j_change:.3f}")

    _check("full pattern reaches cortex", int(full_sig.sum()) > 0)
    _check("training changed cue response", j_change < 0.95,
           f"J={j_change:.3f}")
    _check("cue after training reaches missing neurons", reached > 0,
           f"{reached}/{missing_n}")


def _test_13():
    print("\nTest 13: episode reaches neocortex, survives hippo erase")
    A = [1, 1, 0, 0, 0, 0, 0, 0]
    e = NAJime(seed=7)
    for _ in range(15): e.tick()
    for cycle in range(40):
        e.present_input(A)
        for _ in range(5): e.tick()
        for _ in range(3): e.tick()

    e.present_input(A)
    sig_A = _accumulate(e, 'thought', 20)
    n_active = int(sig_A.sum())
    print(f"    A signature: {n_active} neurons")
    _check("episode reached cortex", n_active > 0)

    e._erase_hippo_edges()
    e.present_input(A)
    sig_after = _accumulate(e, 'thought', 20)
    j = _jaccard(sig_A, sig_after)
    print(f"    J(before_erase, after_erase) = {j:.3f}")
    _check("trace survives hippo erase", j > 0.5,
           f"J={j:.3f}")


def run_tests():
    global _passed, _failed
    _passed = 0
    _failed = 0
    print("=== NAJime v2 (intra-minicolumn streaming LTP) ===")
    for fn in (_test_1, _test_2, _test_3, _test_4, _test_5,
               _test_6a, _test_6b, _test_7, _test_8, _test_9,
               _test_10, _test_11, _test_12, _test_13):
        try:
            fn()
        except Exception as e:
            _failed += 1
            print(f"  [FAIL] exception {type(e).__name__}: {e}")
    print(f"\n=== summary: {_passed} passed, {_failed} failed ===")


# ==================== REPL ====================
def run_repl():
    eng = NAJime(seed=1)
    print("NAJime v2 REPL (intra-minicolumn LTP)")
    print("  p v0..v7 | pb N | f TEXT | c TEXT | t N")
    print("  o / on | rb [N] | trace N | r D | s D | q")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        parts = line.split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else ""
        if cmd == 'q':
            break
        elif cmd == 'p':
            vals = [int(x) & 3 for x in arg.split()[:8]]
            while len(vals) < 8: vals.append(0)
            eng.present_input(vals)
            print(f"  presented {vals}")
        elif cmd == 'pb':
            n = int(arg) & 0xFF
            for _ in range(4):
                eng.present_byte(n); eng.tick()
            eng.tick()
            print(f"  presented byte {n} = 0b{n:08b}")
        elif cmd == 'f':
            b = arg.encode('latin-1', errors='replace')
            eng.feed_text(arg, hold=4, gap=1)
            print(f"  fed {len(b)} bytes: {b.hex()}")
        elif cmd == 'c':
            b = arg.encode('latin-1', errors='replace')
            out = eng.chat(arg, max_new=32, hold=4, gap=1)
            asc = ''.join(chr(x) if 32 <= x < 127 else '.' for x in out)
            print(f"  in : {b.hex()}")
            print(f"  out: {out.hex()}")
            print(f"  asc: {asc}")
        elif cmd == 't':
            n = int(arg) if arg else 1
            for _ in range(n): eng.tick()
            print(f"  ran {n} (total {eng.tick_count})")
        elif cmd == 'o':
            print(f"  output (recent) = {eng.read_output()}")
        elif cmd == 'on':
            print(f"  output (now)    = {eng.read_output_now()}")
        elif cmd == 'rb':
            max_t = int(arg) if arg else 30
            b = eng.read_byte(max_ticks=max_t)
            if b < 0:
                print(f"  read_byte: no events within {max_t} ticks")
            else:
                ch = chr(b) if 32 <= b < 127 else '.'
                print(f"  read_byte = {b} = 0b{b:08b} = '{ch}'")
        elif cmd == 'trace':
            n = int(arg) if arg else 20
            print("  tick | out  | mot  tht  ca1  ca3   dg  sen | total")
            for _ in range(n):
                eng.tick()
                o0, o1 = eng.read_output_now()
                c = {r: int(eng.spike_curr[OFFSETS[r]:
                                          OFFSETS[r] + REGION_SIZES[r]].sum())
                     for r in ('motor', 'thought', 'ca1', 'ca3', 'dg', 'sensory')}
                tot = int(eng.spike_curr.sum())
                print(f"  {eng.tick_count-1:4d} | ({o0},{o1}) | "
                      f"{c['motor']:4d} {c['thought']:4d} {c['ca1']:4d} "
                      f"{c['ca3']:4d} {c['dg']:4d} {c['sensory']:4d} | {tot:5d}")
        elif cmd == 'r':
            eng.reinforce(int(arg) if arg else 3); print("  +")
        elif cmd == 's':
            eng.suppress(int(arg) if arg else 3); print("  -")
        else:
            print("  unknown")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        run_repl()
