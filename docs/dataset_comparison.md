# Dataset research (draft, 21 Nov 2026)

Rule used: only numbers read from a source page are in the table. Anything not confirmed is marked "not verified".

| Dataset | Images | Labels | Camera / resolution | Access | Notes |
|---|---|---|---|---|---|
| APTOS 2019 (Kaggle) | 3,662 train + 1,928 test (test labels not public) | DR grade 0-4 by a clinician | "variety of imaging conditions" | Kaggle competition, accept rules; 10.21 GB | Training set. Our EDA: grade 0 = 49%; image size predicts grade (source confounding). Permitted use: read the competition rules (not verified). |
| EyePACS / Kaggle DR 2015 | train 35,126, validation 10,906, test 42,670 (TFDS split) | 5 classes | varied; original 89.15 GiB; smaller TFDS configs 1.30-3.96 GiB | Manual download from Kaggle required | Largest, too big for a one-day build. Future work. |
| IDRiD | 516 (413 train, 103 test) | DR grade 0-4, DME grade 0-2; 81 images with lesion masks | Kowa VX-10 alpha, 50 degrees, 4288x2848 JPG | IEEE DataPort, DOI 10.21227/H25W98, CC BY 4.0 | One eye clinic in Nanded, Maharashtra, India (2009-2017). Used as the external test set. |
| Messidor-2 | 1,748 images (874 examinations) | The provider states it has no DR ground truth | Topcon TRC NW6 (extension part); PNG and JPG | Free for research and education, form required, no redistribution | Not usable out of the box. |
| SUSTech-SYSU | 1,219 | Grades 0-4 plus laser category; only 24 grade-1 images | Topcon TRC-50DX, 2880x2136 | figshare doi 10.6084/m9.figshare.12570770.v1 | Few grade 1 and 4 examples. |
| FGADR | 2,842 | Grades 0-4; lesion masks for 1,842 | not specified in the paper | Authors say it will be released at csyizhou.github.io/FGADR | Access not verified. |
| DDR | not verified | not verified | not verified | CC BY 4.0 on a Hugging Face mirror | Mirror pages disagree on the image count, so no figure is quoted. |

## Decision
Train on APTOS 2019 (free, 5 grades, right size for one day, already analysed). Test on IDRiD (different clinic and camera, India, same 0-4 scale) as external validation.

## Sources
- IDRiD: https://www.mdpi.com/2306-5729/3/3/25
- Messidor-2 (ADCIS): https://www.adcis.net/en/third-party/messidor2/
- EyePACS in TensorFlow Datasets: https://www.tensorflow.org/datasets/catalog/diabetic_retinopathy_detection
- APTOS 2019 counts and size: https://academictorrents.com/details/d8653db45e7f111dc2c1b595bdac7ccf695efcfd
- SUSTech-SYSU: https://www.nature.com/articles/s41597-020-00755-0
- FGADR: https://arxiv.org/abs/2008.09772
- DDR (mirror): https://huggingface.co/datasets/ctmedtech/DDR-dataset
