#!/usr/bin/env python3
"""Download the selected Thingi10K STLs and write a manifest.csv."""
import csv, json, os, struct, time, urllib.request, urllib.error

BASE = "https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main/raw_meshes"
OUT  = "resources"
os.makedirs(OUT, exist_ok=True)

sel = json.load(open('.thingi-meta/selection.json'))
# stable, self-documenting order
order = ["genus0_solid","genus1","genus2","genus3_5","genus6_11","genus12plus",
         "multi_2_3","multi_4_9","multi_10plus","open_surface","nonmanifold","selfintersect"]
sel.sort(key=lambda it: (order.index(it['bucket']), -it['mech'], it['faces']))

def fetch(fid, dst, tries=4):
    url = f"{BASE}/{fid}.stl"
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"convex-decomp-testset/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            with open(dst, "wb") as f: f.write(data)
            return len(data)
        except urllib.error.HTTPError as e:
            if e.code in (429,503) and a < tries-1:
                time.sleep(2*(a+1)); continue
            raise
        except Exception:
            if a < tries-1: time.sleep(1.5*(a+1)); continue
            raise
    return 0

def stl_tris(path):
    """Return (kind, triangle_count) verifying the STL parses."""
    with open(path,"rb") as f: head = f.read(84)
    if head[:5].lower() == b"solid" and b"facet" in open(path,"rb").read(512).lower():
        n = sum(1 for ln in open(path,"rb") if ln.strip().lower().startswith(b"facet"))
        return ("ascii", n)
    if len(head) >= 84:
        n = struct.unpack("<I", head[80:84])[0]
        return ("binary", n)
    return ("unknown", 0)

rows = []
ok = 0
for i, it in enumerate(sel, 1):
    fid = it['file_id']; dst = os.path.join(OUT, f"{fid}.stl")
    try:
        nbytes = fetch(fid, dst)
        kind, ntri = stl_tris(dst)
        ok += 1
        it2 = dict(it); it2.update(bytes=nbytes, stl_kind=kind, stl_tris=ntri)
        rows.append(it2)
        print(f"[{i:3}/100] {it['bucket']:14} {fid:>8}.stl  {kind:6} {ntri:>7} tris  {nbytes/1024:6.0f} KB  {it['name'][:34]}")
    except Exception as e:
        print(f"[{i:3}/100] FAILED {fid}: {e}")

# manifest
cols = ["file_id","bucket","genus","comps","faces","verts","closed","vertex_manifold",
        "edge_manifold","oriented","self_intersections","boundary_edges","solid",
        "euler","stl_kind","stl_tris","bytes","category","subcategory","name","tags",
        "license","thing_id"]
with open(os.path.join("resources","manifest.csv"),"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for r in rows: w.writerow(r)

print(f"\nDownloaded {ok}/100 STLs -> {OUT}/")
print(f"Manifest -> resources/manifest.csv")
tot = sum(r['bytes'] for r in rows)
print(f"Total size: {tot/1e6:.1f} MB")
