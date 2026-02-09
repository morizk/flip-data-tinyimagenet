# Fix GitHub Push Authentication

The issue is that you're authenticated as "Morizk-witco" but need to push as "morizk".

## Solution: Use Personal Access Token

1. **Create a Personal Access Token:**
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Name: "flip-data-repo"
   - Select scope: `repo` (full control)
   - Click "Generate token"
   - **Copy the token immediately** (you won't see it again)

2. **Push with token:**
   ```bash
   git push -u origin main
   ```
   When prompted:
   - **Username**: `morizk`
   - **Password**: `<paste your personal access token here>`

## Alternative: Update Git Credentials

If you want to update your stored credentials:

```bash
# Remove old credentials
git config --global --unset credential.helper
git credential-cache exit

# Or manually edit:
# ~/.git-credentials
```

Then push again and enter your token when prompted.

## Verify After Push

```bash
git remote -v
git log --oneline -1
```

The repository should now be accessible at:
https://github.com/morizk/flip-data-tinyimagenet

