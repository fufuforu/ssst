# S1 anchor -> proposal -> support -> grouping decomposition (val32, context)

A/B/C are 2D projected CENTRE coverage proxies, not true 3D object coverage: a Gaussian footprint can cover an object even when no centre projects inside it

## Coverage by size

| size | GT count | A all1024 | B FPS100 | C local8 | D recovered |
|---|---|---|---|---|---|
| all | 163 | 0.908 | 0.804 | 0.908 | 0.037 |
| small | 26 | 0.538 | 0.231 | 0.538 | 0.000 |
| medium | 61 | 0.951 | 0.803 | 0.951 | 0.016 |
| large | 76 | 1.000 | 1.000 | 1.000 | 0.066 |

## Conditional final recall

- P(D) = 0.037
- P(D|A) = 0.041 (n=148)
- P(D|B) = 0.046 (n=131)
- P(D|C) = 0.041 (n=148)
- P(D|A=1,B=0) = 0.000

## Transitions

```
{
  "A=0": 15,
  "A=1,B=0": 17,
  "B=1,C=1": 131,
  "A=1,C=0": 0,
  "C=1,D=0": 142,
  "C=1,D=1": 6,
  "A=0,D=1": 0
}
```

C=1,D=0 (local8 covers the GT but grouping fails): 142 (0.871 of all GT, 0.959 of C-covered GT).
A=1,B=0 (evidence exists but FPS dropped it): 17 (0.104 of all GT).
