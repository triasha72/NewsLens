# Local data

Dataset files are intentionally excluded from version control.

NewsLens will use the Microsoft MIND-small dataset under its applicable research
license. Downloaded archives, extracted news records, and behavior logs must not
be committed to GitHub.

Expected local structure:

```text
data/
  MINDsmall_train/
    news.tsv
    behaviors.tsv
  MINDsmall_dev/
    news.tsv
    behaviors.tsv
```

Validated records can be materialized into a local DuckDB database with:

```bash
python -m newslens build-warehouse \
  --data-dir data \
  --split train \
  --output warehouses/mindsmall_train.duckdb
```

The `warehouses/` directory and `*.duckdb` files are ignored because the
database contains data derived from the licensed source files. See
[`docs/DATA_WAREHOUSE.md`](../docs/DATA_WAREHOUSE.md) for the schema and SQL
examples.
