# Notebooks

Use notebooks for exploratory data analysis and visual checks only.

Reusable parsing, feature, model, and evaluation logic belongs in
`src/newslens/` and must be covered by tests. A result that exists only in a
notebook is not considered part of the reproducible pipeline.
