# Start here

## 1. Verify the starter

Run the setup command for your operating system from the repository root.
Confirm that:

- Python reports version 3.12;
- both tests pass;
- Ruff reports no errors;
- the `newslens` command prints the next milestone.

Do not continue if a different Python environment is active.

## 2. Publish the initial checkpoint

Create an empty public GitHub repository named `newslens`, then run:

```bash
git init
git branch -M main
git add .
git commit -m "chore: initialize NewsLens package and test environment"
git remote add origin https://github.com/triasha72/newslens.git
git push -u origin main
```

If the remote repository already contains a README or license, clone it first,
copy this starter's contents into the clone, and commit from there. Do not copy a
different repository's `.git` directory.

## 3. Begin the first feature branch

After the initial checkpoint is visible on GitHub:

```bash
git switch -c feat/mind-data-loader
```

The first implementation task will be to define typed article and impression
records and validate the MIND `news.tsv` and `behaviors.tsv` schemas.

## Evidence to retain

For every milestone, record:

- the question being solved;
- the assumptions you made;
- the tests you added;
- the result and limitations;
- the commit or pull-request link.

Use `docs/LEARNING_LOG.md` and `docs/DECISIONS.md` while building.
