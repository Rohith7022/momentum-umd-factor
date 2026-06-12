# Momentum (UMD) Factor Construction
## VCU – Advanced Financial Analytics (FIRE 691)

Implementation of the **Carhart (1997) Momentum Factor (UMD – Up Minus Down)** as an extension to the Fama-French 3-Factor Model, built from scratch using CRSP monthly return data spanning 1962–2025.

---

## Methodology

**Formation window**: Cumulative returns from month t-12 to t-2 (skipping t-1 to avoid short-term reversal)

Portfolio sorts:
- Split firms into Small/Big using NYSE median market cap at month t-1
- Within each size group, sort into High/Neutral/Low using 30th/70th NYSE percentile breakpoints
- 6 portfolios: SH, SN, SL, BH, BN, BL (value-weighted)

**MOM = (SH + BH)/2 - (SL + BL)/2**

---

## Analysis

1. MOM factor construction and visualization
2. Momentum decile portfolio returns across 1, 12, and 36-month horizons
3. Netflix (NFLX, PERMNO 89393) momentum decile tracking over time
4. Alternative formation window experiments

---

## Results

| Strategy | Monthly Mean | Sharpe Ratio |
|----------|-------------|--------------|
| Standard (12,2) | 0.583%/mo | 0.507 |
| My Factor (9,2) | 0.459%/mo | 0.412 |
| No-skip (12,1) | 0.471%/mo | 0.388 |
| Short (6,2) | 0.368%/mo | 0.351 |
| Long (18,2) | 0.382%/mo | 0.336 |

- MOM annualized mean: **7.00%** | Annualized Std: **13.80%** | Sharpe: **0.507**
- Final output: `my_ff4_factors.csv` with 708 monthly observations (July 1964 onward)

---

## Tech Stack

`Python` `pandas` `numpy` `matplotlib` `Google Colab` `WRDS/CRSP`

---
*Virginia Commonwealth University - MS Business (Financial Analytics) - FIRE 691*
