from tqdm import tqdm   # works in notebooks (auto‑detects)

N, M, P = 100, 50, 20            # example sizes

for i in tqdm(range(N), desc='outer'):
    for j in tqdm(range(M), desc='mid',   leave=False, position=1):
        for k in tqdm(range(P), desc='inner', leave=False, position=2):
            # <-- your work here -->
            pass