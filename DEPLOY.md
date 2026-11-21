# Putting Netra Scan online (GitHub + Hugging Face Spaces)

Do this only after `python scripts/02_train_models.py` has made `models/netra_bundle.joblib`,
and `python tests/test_app.py` passes on your laptop.

## 1. GitHub (stores the code)

1. On github.com click **New repository**. Name it `netra-scan`. Choose Public. Do **not** tick "Add a README".
2. In the VS Code terminal, inside the project folder (replace `YOUR-NAME`):

```
git init
git add .
git commit -m "Netra Scan: retina screening assistant"
git branch -M main
git remote add origin https://github.com/YOUR-NAME/netra-scan.git
git push -u origin main
```

If Git asks you to sign in, use the browser window it opens. If it complains that the model file is
bigger than 100 MB, run step 02 again with `--trees 120`.

## 2. Hugging Face Space (runs the website, free)

1. Sign in at huggingface.co, click your picture, **New Space**. Name `netra-scan`, SDK **Docker** (blank template), CPU basic (free), Public.
2. Create an access token: Settings > Access Tokens > New token, type **Write**. Copy it.
3. In a NEW folder (not inside your project), clone the empty Space (replace `YOUR-HF-NAME`; when asked for a password paste the token):

```
git clone https://huggingface.co/spaces/YOUR-HF-NAME/netra-scan hf-space
```

4. Copy these from your project into `hf-space`: the folders `app`, `netra`, `models`, and the files `Dockerfile` and `requirements-deploy.txt`. Copy `deploy/HF_SPACE_README.md` into `hf-space` and rename it to `README.md`.
5. The model file is big, so Hugging Face needs Git LFS for it. Inside `hf-space`:

```
git lfs install
git lfs track "models/*.joblib"
git add .
git commit -m "Deploy Netra Scan"
git push
```

6. Open your Space page. The **Logs** tab shows the build (the first build takes about 10 minutes). When it says "Running", the site is live at
   `https://YOUR-HF-NAME-netra-scan.hf.space`.
7. Test: open the site, upload a photo, check "How well does this model work?".

## If something fails

* Build error in Logs: copy the last 20 lines and send them to me.
* "Model not loaded" on the site: `models/netra_bundle.joblib` or `models/pins.txt` was not copied.
* Free Spaces go to sleep after a while; the first visit afterwards takes a minute to wake up.
