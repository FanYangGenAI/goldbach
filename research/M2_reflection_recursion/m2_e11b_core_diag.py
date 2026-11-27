"""M2-E11b（探索性，非预注册）：0 的连通块之外的小顶点 [−√q, √q) 属于什么样的连通块。"""
import sys, json, math, random
from collections import defaultdict, deque, Counter
from m2_e4_random_model import bits_needed
q, eta, S, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
Z = round(q ** (1 + eta)); B = bits_needed(Z, q); rng = random.Random(seed); V = math.isqrt(q)
sizes = Counter(); deg0 = 0; tot_out = 0; parity = Counter(); side = Counter()
for _ in range(S):
    y = rng.getrandbits(B)
    adj = defaultdict(list); adj[0].append(-1); adj[-1].append(0)
    for d in range(2, Z + 1):
        e = y % d; adj[e].append(e - d); adj[e - d].append(e)
    comp = {}; cid = 0
    for v in range(-V, V):
        if v in comp: continue
        dq = deque([v]); comp[v] = cid; members = [v]
        while dq:
            x = dq.popleft()
            for w in adj[x]:
                if w not in comp: comp[w] = cid; dq.append(w); members.append(w)
        if cid != comp[0] if 0 in comp else True:
            pass
        cid += 1
        if 0 not in members:
            sizes[min(len(members), 10**6)] += 1
    c0 = comp[0]
    for v in range(-V, V):
        if comp[v] != c0:
            tot_out += 1
            if not adj[v]: deg0 += 1
            parity[(y - v) % 2] += 1   # 顶点 v 对应整数 y − v
            side["A(v>=0)" if v >= 0 else "B(v<0)"] += 1
small = sorted(sizes.items())[:8]
print(json.dumps(dict(q=q, eta=eta, S=S, out_small_vertices=tot_out, isolated_among_out=deg0,
      parity_of_integer=dict(parity), side=dict(side), other_comp_sizes_head=small,
      n_other_comps=sum(sizes.values()), big_other=sum(c for s, c in sizes.items() if s > 50))))
