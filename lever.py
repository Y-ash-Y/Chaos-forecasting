import sys

def main():
    data = list(map(int, sys.stdin.read().strip().split()))
    it = iter(data)
    t = next(it)
    out_lines = []
    for _ in range(t):
        n = next(it)
        a = [next(it) for _ in range(n)]
        b = [next(it) for _ in range(n)]
        D = 0
        for i in range(n):
            if a[i] > b[i]:
                D += (a[i] - b[i])
        out_lines.append(str(D + 1))
    print("\n".join(out_lines))

if __name__ == "__main__":
    main()
