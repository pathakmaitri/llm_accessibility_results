# LLM Accessibility Research — Analysis Script

This script analyzes how well three LLMs (GPT, Claude, and Gemini) perform at detecting and fixing web accessibility violations in dashboard code. It runs a set of statistical analyses and produces charts comparing model performance across two prompt strategies.

## Requirements

```bash
pip3 install pandas matplotlib scipy numpy openpyxl
```

## Setup

The Excel data file must be in the same folder as `analysis.py`:

```
your-folder/
├── analysis.py
└── llm_accessibility_results.xlsx
```

Running the script will create a `results/` folder automatically.

## Usage

```bash
python3 analysis.py
```

All output goes to `results/analysis_results.txt` rather than the terminal. A single confirmation line prints when the script finishes.

## Input Format

The script expects an Excel file with the following columns:

| Column | Description |
|---|---|
| `Model` | GPT, Claude, or Gemini |
| `Prompt` | P1 or P2 |
| `Dashboard_ID` | Identifier for the dashboard being evaluated |
| `Issue_Type` | The accessibility violation type |
| `False_Positives` | Score 0-3 |
| `Detection_Accuracy` | Score 0-3 |
| `Implementation_Accuracy` | Score 0-3 |
| `Code_Reasoning` | Score 0-3 |
| `Violation_Presence` | Score 0-3 |
| `Total_Score` | Sum of the five metrics (auto-calculated if missing) |

Each metric is scored 0 to 3, giving a maximum total score of 15.

## Analyses

1. **Model performance** — mean total score per model, per-metric breakdown, and a one-way ANOVA
2. **Prompt comparison** — mean scores for P1 vs P2, percentage change, and a paired t-test
3. **Success rate** — how often each model and prompt achieved a perfect score of 15
4. **Error analysis** — failure counts per metric per model (score of 0 or 1)
5. **Issue-type breakdown** — mean scores cross-tabulated by violation type and model
6. **Violation comparison** — mean scores per violation type and an independent t-test

## Charts

All charts are saved to the `charts/` folder at 150 DPI.

| File | Description |
|---|---|
| `chart1_model_comparison.png` | Bar chart of mean total score per model with standard deviation error bars |
| `chart2_prompt_comparison.png` | Grouped bar chart comparing P1 and P2 scores for each model |
| `chart4_spider_charts.png` | Radar charts showing each model's average score across all five metrics |
| `chart5_stacked_errors.png` | Stacked bar chart of failure counts per metric per model |

## Configuration

A few constants at the top of `analysis.py` can be adjusted if needed:

```python
EXCEL_FILE    = "llm_accessibility_results.xlsx"
OUTPUT_FOLDER = "results"
PERFECT_SCORE = 15
```