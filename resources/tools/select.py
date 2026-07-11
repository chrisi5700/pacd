#!/usr/bin/env python3
"""Deterministically pick ~100 topology-varied Thingi10K meshes for a convex
decomposition test corpus. Selection is driven by topology (genus, components,
manifoldness, boundary) and biased toward mechanical/CAD-like parts via tags."""
import csv, collections, json

def num(x):
    try: return float(x)
    except Exception: return None

def truthy(x): return str(x).strip().lower() in ("true", "1", "yes")

# ---- load metadata -------------------------------------------------------
G = {}
with open('.thingi-meta/geometry_data.csv') as f:
    for r in csv.DictReader(f): G[r['file_id']] = r
S = {}
with open('.thingi-meta/input_summary.csv') as f:
    for r in csv.DictReader(f): S[r['ID']] = r
C = {}
with open('.thingi-meta/contextual_data.csv') as f:
    for r in csv.DictReader(f): C[r['Thing ID']] = r
TAGS = collections.defaultdict(set)
with open('.thingi-meta/tag_data.csv') as f:
    for r in csv.DictReader(f): TAGS[r['Thing ID']].add(r['Tag'].strip().lower())

MECH = ("bracket","mount","gear","holder","clip","clamp","bearing","flange",
        "pulley","hinge","screw","bolt","nut","adapter","spacer","standoff",
        "coupler","coupling","gimbal","frame","plate","knob","valve","fitting",
        "hook","roller","washer","gasket","enclosure","housing","rail","wheel",
        "sprocket","threaded","thread","fixture","jig","mechanical","machine",
        "cnc","router","extruder","nozzle","axle","shaft","bushing","lever",
        "arm","joint","linkage","gearbox","motor","servo","actuator","pipe",
        "tube","connector","socket","hex","spanner","wrench","tool","stand")

def info(fid):
    g = G[fid]; s = S.get(fid, {}); c = C.get(s.get('Thing ID',''), {})
    faces = num(g['num_faces']); verts = num(g['num_vertices'])
    comps = int(num(g['num_connected_components']) or 0)
    euler = num(g['euler_characteristic'])
    be    = num(g['num_boundary_edges']) or 0
    vm = truthy(g['vertex_manifold']); em = truthy(g['edge_manifold'])
    ori = truthy(g['oriented']); solid = truthy(g['solid'])
    si  = num(g['num_self_intersections']) or 0
    degen = (num(g['num_geometrical_degenerated_faces']) or 0) + \
            (num(g['num_combinatorial_degenerated_faces']) or 0)
    closed = (be == 0)
    manifold = vm and em
    # genus reliable only for closed, manifold, oriented
    genus = None
    if closed and manifold and ori and euler is not None and comps > 0:
        h = comps - euler/2.0
        if abs(h - round(h)) < 1e-6: genus = int(round(h))
    tags = TAGS.get(s.get('Thing ID',''), set())
    name = c.get('Name','') or ''
    sub  = (c.get('Sub-category','') or '')
    blob = (name + ' ' + sub + ' ' + ' '.join(tags)).lower()
    mech = sum(1 for k in MECH if k in blob)
    return dict(file_id=fid, thing_id=s.get('Thing ID',''), name=name,
                category=c.get('Category',''), subcategory=sub,
                faces=int(faces or 0), verts=int(verts or 0), comps=comps,
                euler=int(euler) if euler is not None else None, genus=genus,
                closed=closed, vertex_manifold=vm, edge_manifold=em,
                oriented=ori, self_intersections=int(si), solid=solid,
                degenerate=int(degen), boundary_edges=int(be),
                license=c.get('License','') or s.get('License',''),
                tags=';'.join(sorted(tags)), mech=mech)

items = [info(fid) for fid in G]

# reasonable size window: skip 4-face degenerates and multi-million monsters
def sane(it):
    return 300 <= it['faces'] <= 200000 and it['degenerate'] == 0

# ---- buckets: (name, predicate, target) ---------------------------------
def clean(it):  # well-formed closed manifold solid
    return it['closed'] and it['vertex_manifold'] and it['edge_manifold'] and it['oriented']

buckets = [
    ("genus0_solid",   lambda it: clean(it) and it['comps']==1 and it['genus']==0, 14),
    ("genus1",         lambda it: clean(it) and it['comps']==1 and it['genus']==1, 12),
    ("genus2",         lambda it: clean(it) and it['comps']==1 and it['genus']==2, 8),
    ("genus3_5",       lambda it: clean(it) and it['comps']==1 and it['genus'] in (3,4,5), 10),
    ("genus6_11",      lambda it: clean(it) and it['comps']==1 and it['genus'] is not None and 6<=it['genus']<=11, 8),
    ("genus12plus",    lambda it: clean(it) and it['comps']==1 and it['genus'] is not None and it['genus']>=12, 8),
    ("multi_2_3",      lambda it: clean(it) and it['comps'] in (2,3), 8),
    ("multi_4_9",      lambda it: clean(it) and 4<=it['comps']<=9, 6),
    ("multi_10plus",   lambda it: clean(it) and it['comps']>=10, 4),
    ("open_surface",   lambda it: (not it['closed']) and it['vertex_manifold'] and it['edge_manifold'], 8),
    ("nonmanifold",    lambda it: not (it['vertex_manifold'] and it['edge_manifold']), 8),
    ("selfintersect",  lambda it: it['closed'] and it['vertex_manifold'] and it['edge_manifold'] and it['self_intersections']>0, 6),
]

pool = [it for it in items if sane(it)]
chosen = {}
report = []
for bname, pred, target in buckets:
    cands = [it for it in pool if it['file_id'] not in chosen and pred(it)]
    # prefer mechanical parts, then a mid-range face count for variety
    cands.sort(key=lambda it: (-it['mech'], abs(it['faces']-6000)))
    picked = cands[:target]
    for it in picked:
        it['bucket'] = bname
        chosen[it['file_id']] = it
    report.append((bname, len(picked), target))

sel = list(chosen.values())
print("=== bucket fill (got/target) ===")
for b,g,t in report: print(f"  {b:16} {g:2}/{t}")
print(f"TOTAL selected: {len(sel)}")
mechn = sum(1 for it in sel if it['mech']>0)
print(f"with mechanical tag/name hit: {mechn}/{len(sel)}")
tf = sum(it['faces'] for it in sel)
print(f"face counts: min={min(it['faces'] for it in sel)} max={max(it['faces'] for it in sel)} total={tf}")

with open('.thingi-meta/selection.json','w') as f:
    json.dump(sel, f, indent=0)
print("wrote .thingi-meta/selection.json")
