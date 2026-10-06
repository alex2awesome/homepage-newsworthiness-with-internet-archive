# homepage-newsworthiness-with-internet-archive

Code for *NewsHomepages*, a study of how news homepages encode editorial prioritization, or "newsworthiness", decisions. Editors place the stories they judge most important higher, further left and in larger cards. We use about three years of twice-daily captures of more than 3,000 news homepages (the `news-homepages` collection on the Internet Archive) to (1) parse each homepage into article cards with exact positions and sizes, (2) train pairwise models that predict which of two articles an outlet would display more prominently, and (3) apply those models to rank other text, such as San Francisco city-council policies, by newsworthiness. The research question is whether the prioritization decisions behind homepage layouts can be learned at scale and transferred across outlets and domains. Work by Alexander Spangher with Ben Welsh, Arda Kaz, Michael Vu and Naitian Zhou.

## Related paper

"NewsHomepages: Homepage Layouts Capture Information Prioritization Decisions" (Welsh, Kaz, Vu, Zhou, Spangher). The ACL-style draft is in `latex/naacl2024/acl_latex.tex`. Data collection runs through https://github.com/palewire/news-homepages and is archived at https://archive.org/details/news-homepages.

## Layout

- `scripts/homepage_parsing/` -- Playwright DOM parser that extracts article bounding boxes from saved homepage HTML (`get_bounding_boxes_from_html.py`); Internet Archive downloader (`server-script.py`); CSV-to-COCO conversion and a Detectron2 script for a screenshot-based layout detector; a BERT "is this an article?" classifier.
- `scripts/clean_training_data/` -- cleans extracted link text with an LLM through vLLM (`run_prompts.py`).
- `scripts/ranking_articles/joint_ranking.py` -- turns pairwise-model outputs into full rankings per outlet.
- `notebooks/` -- `2024-05-30__demo.ipynb` walks through download and parsing; others analyze outlet-vs-outlet agreement and city-council rankings.
- `latex/` -- paper draft and figures.
- `data/` -- local data, gitignored.

## How to run

1. `pip install -r scripts/homepage_parsing/requirements.txt` plus `playwright`, `internetarchive` and `pandas`.
2. Download captures: `python scripts/homepage_parsing/server-script.py --download_html --start_idx 0 --end_idx 10`.
3. Parse layouts: follow `notebooks/2024-05-30__demo.ipynb`, which calls `get_bounding_boxes_from_html`.
4. Optional: `bash scripts/homepage_parsing/trainingscript.sh` trains the layout detector; `python scripts/ranking_articles/joint_ranking.py` produces rankings.

## Data

Homepage HTML, screenshots and link metadata come from the public `news-homepages` Internet Archive collection; `data/sites.csv` lists the outlets. Derived data (bounding-box CSVs, pairwise comparison sets, GPT summaries of city-council policies, rankings) is generated into `data/` and is not included; it runs to several GB.

## Status

Last substantive changes October-November 2024 (analysis notebooks and paper draft).
