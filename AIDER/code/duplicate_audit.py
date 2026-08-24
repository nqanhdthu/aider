from pathlib import Path
import hashlib
from collections import defaultdict
import imagehash
from PIL import Image

def sha256_file(path, chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def phash64(path):
    with Image.open(path) as im:
        return int(str(imagehash.phash(im,hash_size=8,highfreq_factor=4)),16)

def hamming64(a,b):
    return (a^b).bit_count()

def candidate_pairs_exact_hamming_le4(items):
    """
    Complete candidate generation for a <=4-bit threshold using five disjoint
    bit blocks. Any pair differing in <=4 bits must match in at least one block.
    items: iterable of (record_id, uint64_phash)
    """
    blocks=[(0,13),(13,26),(26,39),(39,52),(52,64)]
    buckets=[defaultdict(list) for _ in blocks]
    hashes={}
    for rid,h in items:
        hashes[rid]=h
        for bi,(lo,hi) in enumerate(blocks):
            mask=(1<<(hi-lo))-1
            buckets[bi][(h>>lo)&mask].append(rid)
    seen=set()
    for bucket in buckets:
        for ids in bucket.values():
            if len(ids)<2: continue
            for i in range(len(ids)):
                for j in range(i+1,len(ids)):
                    a,b=ids[i],ids[j]
                    key=(a,b) if a<b else (b,a)
                    if key in seen: continue
                    seen.add(key)
                    if hamming64(hashes[a],hashes[b])<=4:
                        yield key

