# JustEmbed Quarto documentation (v0.1.1)

## Render to HTML

1. **Install Quarto** (if needed): https://quarto.org/docs/get-started/

   On Ubuntu:
   ```bash
   sudo apt-get install quarto
   ```
   Or download from https://quarto.org/docs/download/

2. **From this directory** (`quarto/`), run:
   ```bash
   quarto render
   ```

3. **Output** is written to `../docs/` (i.e. `v0.1.1/docs/`).

## HTML output paths

After rendering, the site lives under `docs/`:

| Page | Path (relative to repo) |
|------|-------------------------|
| Home | `docs/index.html` |
| Getting Started | `docs/getting-started.html` |
| Guide | `docs/guide.html` |
| Examples | `docs/examples.html` |
| API Reference | `docs/reference.html` |

## GitHub Pages

1. Push the `docs/` folder to your repo.
2. GitHub: **Settings → Pages → Source**: choose **Deploy from a branch**.
3. Branch: `main` (or your default), folder: **/docs**.
4. If you use a custom domain, set it in the same Pages settings.

Your docs will be at `https://<your-username>.github.io/justembed/` or your custom domain.

## Edit and re-render

- Edit any `.qmd` file in `quarto/`.
- Run `quarto render` again from `quarto/`.
- Commit and push the updated `docs/` folder.
