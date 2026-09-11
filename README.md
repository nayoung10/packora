# Packora project page

This directory is the static companion page for the Packora paper. It contains
only HTML, CSS, JavaScript, logos, and paper figures, so it can be published
independently with GitHub Pages. The prediction interface and GPU service live
in `app/`.

The arXiv control links to the published paper, and the Code control links to
the GitHub repository. The Hugging Face control links to the public model repository.
Both **Try Packora** links are populated from the single value in `config.js`:

```javascript
window.PACKORA_APP_URL = "https://packora.example.org";
```

The checked-in value points to the temporary Cloudflare demo. Update it and the
root README demo badge whenever the tunnel URL changes. An empty value keeps
the **Try Packora** controls disabled.
Set it to `http://127.0.0.1:8000` for local integration or to the public HTTPS
app URL when publishing. All page assets use relative paths, so the page works
from a GitHub project subpath without rebuilding.

## Run locally

Serve the project page on its default local port:

```bash
python -m http.server 8001 --directory site
```

Then open <http://127.0.0.1:8001>. Start the Packora application separately as
documented in `app/README.md`; the project-page link will navigate to it.

## GitHub Pages

The workflow in `../.github/workflows/publish-site.yml` automatically copies
`packora-dev/site/` to the `gh-pages` branch in `nayoung10/packora` when `site/`
changes are pushed to `packora-dev/main`. GitHub Pages serves those files at
<https://nayoung10.github.io/packora/>. All editing stays in `packora-dev`;
the public code branch and release exporter are unchanged.

One-time setup:

1. Create a fine-grained GitHub token restricted to `nayoung10/packora`, with
   **Contents: Read and write** permission.
2. Save it in `packora-dev` under **Settings → Secrets and variables → Actions**
   as `PACKORA_PAGES_TOKEN`. Do not put it in files or chat.
3. Run **Actions → Publish project site → Run workflow** to create `gh-pages`.
4. In `packora` under **Settings → Pages**, choose **Deploy from a branch**,
   **gh-pages**, and **/ (root)**, then save.

After setup, commit and push `site/` changes to `main` to publish them. Saving a
local file alone does not trigger GitHub Actions. The workflow can also be run
manually. Renew the token before expiry.

The GPU demo remains on its server. When a temporary Cloudflare tunnel is
recreated, update `site/config.js` and the root README demo badge with the new
URL, then push. This workflow publishes that link; it does not discover new URLs.

The article reproduces only figures included by `paper/main.tex`, with Figure 3
intentionally omitted. The arXiv control uses the official primary logo served
at
<https://arxiv.org/static/base/1.0.1/images/arxiv-logo-primary-light.svg>,
vendored unchanged apart from a trailing newline in `assets/`.

## Verification

```bash
node --check site/project.js
python -m http.server 8001 --directory site
```

The application test suite also checks that this page uses relative assets and
that both **Try Packora** links are driven by `config.js`.
