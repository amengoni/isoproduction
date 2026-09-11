To update your remote Git repository (like GitHub, GitLab, or Bitbucket) with local corrections, follow these standard steps:

### 1. Check Your Status

First, see which files you have changed or added:

```bash
git status

```

### 2. Stage Your Changes

Add the corrected files to the staging area.

* To stage **all** modified and new files:
```bash
git add .

```


* To stage a **specific** file:
```bash
git add path/to/your/file.py

```



### 3. Commit Your Changes

Save your staged changes locally with a descriptive commit message explaining the fixes:

```bash
git commit -m "Fix neutron flux units and apply BIF directly to activation calculation"

```

### 4. Push to the Remote Repository

Upload your local commit(s) to the remote repository.

* If you are working on the default branch (usually `main` or `master`):
```bash
git push origin main

```


* If you are on a feature branch:
```bash
git push origin <your-branch-name>

```



---

### Additional Common Scenarios

#### If you haven't committed yet and want to check your exact changes:

```bash
git diff

```

#### If you modified the last commit locally before pushing:

If you already made a commit locally but added one more quick fix before pushing, you can combine it into the previous commit:

```bash
git add .
git commit --amend --no-edit
git push origin <your-branch-name>

```

#### If remote changes were made while you were working locally:

If someone else pushed code (or you edited code via the web interface), pull the latest updates first to avoid conflicts:

```bash
git pull --rebase origin main
git push origin main

```
