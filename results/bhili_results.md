# Marathi ↔ Dehwali Bhili — results

chrF++ uses `word_order=2`. COPY = echo the source unchanged.
`spurious_copy` = of the items whose reference differs from the source,
the fraction where the model simply echoed the source back.

## chrF++ against the COPY baseline

| test set | direction | COPY | base | fine-tuned | FT − COPY |
|---|---|---|---|---|---|
| GENERAL | mar-bhb | 42.1 | 40.0 | 49.0 | **+6.8** |
| GENERAL | bhb-mar | 40.5 | 40.9 | 60.3 | **+19.8** |
| NUM_NAT | mar-bhb | 46.3 | 44.3 | 52.0 | **+5.7** |
| NUM_NAT | bhb-mar | 44.7 | 45.0 | 65.3 | **+20.6** |
| SHORT_NAT | mar-bhb | 42.6 | 39.3 | 47.1 | **+4.5** |
| SHORT_NAT | bhb-mar | 43.0 | 42.4 | 60.1 | **+17.1** |
| SHORT_HARD | mar-bhb | 33.5 | 31.5 | 40.5 | **+7.0** |
| SHORT_HARD | bhb-mar | 33.9 | 34.1 | 53.8 | **+19.9** |

## Spurious copy rate (lower is better)

| test set | direction | base | fine-tuned |
|---|---|---|---|
| GENERAL | mar-bhb | 41.9% | 8.5% |
| GENERAL | bhb-mar | 48.8% | 2.9% |
| NUM_NAT | mar-bhb | 37.0% | 9.2% |
| NUM_NAT | bhb-mar | 48.5% | 3.0% |
| SHORT_NAT | mar-bhb | 68.4% | 33.3% |
| SHORT_NAT | bhb-mar | 57.9% | 8.8% |
| SHORT_HARD | mar-bhb | 68.4% | 33.3% |
| SHORT_HARD | bhb-mar | 57.9% | 8.8% |

## Numerals

`nsem` is numeral-sequence exact match against the SOURCE. A copying model
scores 100% by construction, so this metric cannot show improvement — it
only shows breakage. The signal is in digit-script match, and the reference's
own preservation rate is the ceiling.

| test set | direction | ref. ceiling | base nsem | FT nsem | base script | FT script |
|---|---|---|---|---|---|---|
| GENERAL | mar-bhb | 85.6% | 100.0 | 100.0 | 67.4 | 75.1 |
| GENERAL | bhb-mar | 91.2% | 99.4 | 100.0 | 75.9 | 78.2 |
| NUM_NAT | mar-bhb | 83.7% | 100.0 | 100.0 | 70.7 | 76.3 |
| NUM_NAT | bhb-mar | 89.7% | 99.1 | 98.8 | 80.1 | 81.0 |
| SHORT_NAT | mar-bhb | 78.6% | 100.0 | 100.0 | 71.4 | 71.4 |
| SHORT_NAT | bhb-mar | 91.7% | 100.0 | 100.0 | 58.3 | 83.3 |
| SHORT_HARD | mar-bhb | 70.0% | 100.0 | 100.0 | 60.0 | 60.0 |
| SHORT_HARD | bhb-mar | 87.5% | 100.0 | 100.0 | 50.0 | 75.0 |
