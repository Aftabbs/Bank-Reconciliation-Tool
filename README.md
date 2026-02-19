# Bank Reconciliation Tool

This project matches rows from your **bank statement** with rows from your **own records** (check register). Both files list the same real transactions, but with different wording and dates. The tool figures out which line in one file goes with which line in the other.
 
--- 

## What’s in this folder

```
Bank Reconcillation Tool/
├── bank_statements.csv       # Bank’s list of transactions (input)
├── check_register.csv        # Your list of transactions (input)
├── config.py                 # Paths and settings (tolerance, thresholds)
├── run_reconciliation.py     # Main script: run matching and optional review
├── requirements.txt          # Python packages to install
├── .gitignore
├── README.md                 # This file
├── src/
│   ├── __init__.py
│   ├── load_data.py          # Read CSVs, normalize types/dates, build term lists
│   ├── unique_amount_matching.py   # Match where amount appears only once
│   ├── ml_matching.py        # Match remaining rows using SVD-style retrieval
│   ├── evaluation.py        # Precision, recall, F1
│   └── workflow.py          # Run matching, load/save validated pairs

```

`run_experiments.py` may exist at the project root; it runs learning-curve and category experiments and writes `experiment_results.json`.

---

## What you need

- Python 3.8 or newer  
- Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Input files

Put these in the project folder:

- **bank_statements.csv** — columns: transaction_id, date, description, amount, type, balance  
- **check_register.csv** — columns: transaction_id, date, description, amount, type, category, notes  

They should describe the **same** set of transactions. Dates and descriptions can differ a bit (e.g. bank says “TRADER JOES”, you wrote “Groceries”).

---

## How to run it

**Run matching and see how many pairs were found and how good they are:**

```bash
python run_reconciliation.py
```

You’ll see how many pairs were proposed, and numbers for precision, recall, and F1. A few example pairs are printed at the end.

**Review pairs one by one (accept or reject), then save and re-run later:**

```bash
python run_reconciliation.py --review
```

Accepted pairs are stored in `validated_pairs.json` and used as “training” next time so the tool can improve on the rest.

**Only evaluate the pairs you’ve already saved (no new matching):**

```bash
python run_reconciliation.py --eval-only
```

---

## What the numbers mean

- **Proposed matches** — How many bank–register pairs the tool suggested.  
- **Correct** — How many of those pairs are actually right (we know from the data setup).  
- **Precision** — Of all proposed pairs, what fraction are correct. High = few wrong suggestions.  
- **Recall** — Of all transactions that should be matched, what fraction did we find. High = we didn’t miss many.  
- **F1** — Single score combining precision and recall; higher is better.

---

## Changing behaviour

Edit **config.py** to adjust:

- How strict amount matching is and how we treat date differences (unique-amount step).  
- How many dimensions we keep in the math step (SVD) and the minimum similarity to suggest a match (ML step).

---

## How we know if a match is “correct”

For this dataset, the correct pair for a bank row is the register row with the same number in the id (e.g. B0084 goes with R0084). That mapping is used **only to compute precision/recall/F1**. The matching logic never looks at ids; it only uses amounts, dates, and the text in descriptions.

--
