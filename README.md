# ViJudge: Benchmarking LLM-as-a-Judge for Vietnamese through Human Preference and Bias Analysis

## Abstract

This repository presents a comprehensive framework for evaluating Large Language Model (LLM) responses in Vietnamese across multiple evaluation dimensions. The study employs both LLM-based automated evaluation and human annotation to assess response quality systematically.

Concretely, this repository implements **ViJudge**, a benchmark and evaluation pipeline covering 10 practical categories, multiple LLM systems, and both pairwise and single-response evaluation settings.

The primary goal of this repository is to provide a **fully reproducible pipeline** for the paper, allowing researchers to replicate experiments, analyze LLM-as-a-Judge reliability under Vietnamese-specific constraints, and understand systematic biases in automated evaluation.

---

## 1. Overview

### Research Objective

This research investigates the reliability and fairness of LLM-based evaluation methods for assessing Vietnamese language responses. The framework integrates:

- **Response Collection**: Gathering LLM responses to Vietnamese questions across diverse categories
- **Automated Evaluation**: Using multiple LLM models as judges (LLM-as-a-Judge) to evaluate response quality
- **Human Annotation**: Implementing a Streamlit-based web interface for human evaluation
- **Bias Analysis**: Systematic analysis of LLM judge biases affecting evaluation reliability
- **Agreement Measurement**: Computing inter-rater agreement metrics between LLM-as-a-Judge and human

### Key Features

- Multi-turn conversation evaluation
- Pairwise and single-model evaluation modes
- Reference-based and reference-free evaluation options
- Comprehensive bias detection framework
- Real-time human annotation with cloud-based vote persistence

---

## 2. Environment Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Google Colab (recommended for execution) or local Python environment. If running locally, some parts of the code (e.g., file paths, environment configurations, or resource settings) may need adjustments.
- Internet connection for API access

### Required API Keys

This project requires API keys from two external service providers:

1. **Groq API Key**: For accessing Groq inference models
   - Register at: https://console.groq.com
   - Generate API key from the dashboard

2. **OpenRouter API Key** (optional, for supplementary models):
   - Register at: https://openrouter.ai
   - Generate API key from settings

### Installation Steps

#### For Google Colab (Recommended)

```bash
# Mount Google Drive (required for data persistence)
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install groq openai python-dotenv pandas

# For human evaluation interface (optional, not required for analysis phases)
!pip install gradio datasets huggingface-hub
```

### API Configuration

Create a `.env` file in the root directory with the following content:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**Important**: The `.env` file must be uploaded to Google Colab before execution. Use the file upload widget provided in each Jupyter notebook.

---

## 3. Directory Structure and Data Organization

### Overall Repository Structure

```
ViJudge/
├── README.md                               # This file
├── prompts/                                # Judge prompt templates
│   └── judge_prompts.jsonl                 # JSONL file containing all template prompts
│
├── phases/                                 # Main execution phases
│   ├── 1_collect_LLMs_responses/           # Phase 1: Data collection
│   │   ├── collect_LLMs_responses.ipynb
│   │   └── prepare_for_LLM-as-a-Judge.ipynb
│   │
│   ├── 2.1_LLM-as-a-Judge/                 # Phase 2.1: Automated evaluation
│   │   ├── judge_for_pair_LLMs.ipynb       # Pairwise model comparison
│   │   └── judge_for_single_LLM.ipynb      # Single model evaluation
│   │
│   ├── 2.2_human_judge/                    # Phase 2.2: Human annotation
│   │   ├── app.py                          # Gradio-based annotation interface
│   │   └── tex-chtml.js                    # MathJax library for LaTeX rendering
│   │
│   ├── 3_LLM_judge_bias_analysis/          # Phase 3: Bias analysis
│   │   ├── language_bias/
│   │   │   ├── language_bias_verdict_generation.ipynb
│   │   │   ├── statistics_language_bias.ipynb
│   │   │   └── visualization_language_bias.ipynb
│   │   │
│   │   ├── position_bias/
│   │   │   ├── position_bias_verdict_generation.ipynb
│   │   │   ├── statistics_position_bias.ipynb
│   │   │   └── visualization_position_bias.ipynb
│   │   │
│   │   └── verbosity_bias/
│   │       ├── verbosity_bias_verdict_generation.ipynb
│   │       ├── statistics_verbosity_bias.ipynb
│   │       └── visualization_verbosity_bias.ipynb
│   │
│   └── 4_measure_agreement/                # Phase 4: Inter-rater agreement
│       ├── agreement.ipynb
│       └── visualization_agreement.ipynb
│
└── data/                                   # Data storage
    ├── questions/
    │   └── questions_2_turns.csv           # Input question dataset
    │
    ├── LLMs_responses/                     # Model responses (generated)
    │   ├── openai_gpt-oss-20b_response.csv
    │   ├── llama-3.3-70b-versatile_response.csv
    │   ├── moonshotai_kimi-k2-instruct_response.csv
    │   ├── qwen_qwen3-32b_response.csv
    │   └── deepseek_deepseek-chat-v3.1_free_response.csv
    │
    ├── datasets_before_judgement/          # Pre-processed datasets
    │   ├── single_openai.csv
    │   ├── single_llama.csv
    │   ├── single_moonshotai.csv
    │   ├── single_qwen.csv
    │   ├── single_deepseek.csv
    │   ├── pair_openai_llama.csv
    │   ├── pair_openai_moonshotai.csv
    │   ├── pair_openai_qwen.csv
    │   ├── pair_openai_deepseek.csv
    │   ├── pair_llama_moonshotai.csv
    │   ├── pair_llama_qwen.csv
    │   ├── pair_llama_deepseek.csv
    │   ├── pair_moonshotai_qwen.csv
    │   ├── pair_moonshotai_deepseek.csv
    │   ├── pair_qwen_deepseek.csv
    │   ├── translated/                     # English-translated versions for language bias
    │   │   └── translated_[file_name].xlsx
    │   └── verbosity/                      # Verbose versions for verbosity bias
    │       └── verbosity_[model_name].csv
    │
    ├── gen_judgement/                      # Judge outputs (generated)
    │   ├── openai/
    │   ├── llama/
    │   ├── moonshotai/
    │   ├── qwen/
    │   └── tinyllama/
    │       ├── [judge_name]Judge_single_[model].csv
    │       └── [judge_name]Judge_pair_[model_a]_[model_b].csv
    │
    ├── language_bias/                      # Language bias evaluation results
    │   ├── [judge_name]_language_bias_for_pair_*.csv
    │   └── statistics/
    │       └── kappa_result.csv
    │
    ├── position_bias/                      # Position bias evaluation results
    │   ├── [judge_name]_position_bias_for_pair_*.csv
    │   └── statistics/
    │       └── bias_result.csv
    │
    ├── verbosity_bias/                     # Verbosity bias evaluation results
    │   ├── [judge_name]_verbosity_bias_for_verbosity_*.csv
    │   └── statistics/
    │       └── bias_by_attempt_and_config_result.csv
    │
    ├── visualization/                      # Final formatted results
    │   ├── language_bias/
    │   │   └── language_avg_kappa_ratio_table.csv
    │   ├── position_bias/
    │   │   └── position_avg_bias_ratio.csv
    │   ├── verbosity_bias/
    │   │   └── verbosity_avg_bias_ratio_table.csv
    │   └── agreement/
    │       ├── agreement_ties.csv
    │       ├── agreement_no_ties.csv
    │       └── agreement_category.csv
    │
    ├── data_for_agreement/                 # Human vs AI agreement analysis
    │   ├── human_judge/
    │   ├── LLM_judge/
    │   │   ├── temp_pair_from_single/      # Storage data before merge to pair from single
    │   │   │   └── [judge_name]/
    │   │   │       └── [judge_name]_pair_[model_a]_[model_b].csv
    │   │   ├── [judge_name]_pair.csv
    │   │   └── [judge_name]_pair_from_single.csv
    │   └── results.csv                     # Agreement results
    │
    └── 1500_LLM_responses/
        └── 1500_LLM_responses.csv          # For the web-based annotation interface
```

---

### Data Organization Guide

#### Input Data (`/data/questions/`)

**File**: `questions_2_turns.csv`

**Format**: CSV with the following columns:
```
question_id          (int) - Unique identifier
category             (str) - Question category
Vietnamese_content   (str) - Question text in Vietnamese
Vietnamese_reference (str) - Reference answer (optional)
turn                 (int) - Turn number (1 or 2)
```

**Setup Instructions**:
1. Place `questions_2_turns.csv` file in `/data/questions/`
2. Ensure CSV encoding is UTF-8

#### Generated Model Responses (`/data/LLMs_responses/`)

**Format**: CSV generated by Phase 1
```
question_id          (int)
category             (str)
content_turn1        (str)
response_turn1       (str)
content_turn2        (str)
response_turn2       (str)
```

**Models included**:
- `openai_gpt-oss-20b_response.csv`
- `llama-3.3-70b-versatile_response.csv`
- `moonshotai_kimi-k2-instruct_response.csv`
- `qwen_qwen3-32b_response.csv`
- `deepseek_deepseek-chat-v3.1_free_response.csv`

**Location**: Automatically created by Phase 1 notebooks

---

#### Pre-processed Datasets (`/data/datasets_before_judgement/`)

**Single-model files** (5 files):
- `single_openai.csv`, `single_llama.csv`, `single_moonshotai.csv`, `single_qwen.csv`, `single_deepseek.csv`

**Pairwise files** (10 combinations):
- `pair_openai_llama.csv`, `pair_openai_moonshotai.csv`, `pair_openai_qwen.csv`, `pair_openai_deepseek.csv`
- `pair_llama_moonshotai.csv`, `pair_llama_qwen.csv`, `pair_llama_deepseek.csv`
- `pair_moonshotai_qwen.csv`, `pair_moonshotai_deepseek.csv`
- `pair_qwen_deepseek.csv`

**Format** (Standardized for both Single and Pairwise):
```
question_id, category, model_a, model_b, 
content_turn1, reference_turn1, response_turn1_a, response_turn1_b, 
content_turn2, reference_turn2, response_turn2_a, response_turn2_b
```
*(For single-model datasets, `model_b`, `response_turn1_b`, and `response_turn2_b` contain NaN values)*

**Subdirectories**:
- `/translated/` - English-translated CSV files for language bias analysis
- `/verbosity/` - Verbose/repetitive versions of responses for verbosity bias analysis

**Creation**: Generated by `prepare_for_LLM-as-a-Judge.ipynb`

---

#### Judge Outputs (`/data/gen_judgement/`)

**Organization**: One subdirectory per judge model (openai, llama, moonshotai, qwen, tinyllama)

*(Note that you need to run this for each judge model. For each judge model, you need to configure `model_a` and `model_b` in separate files in the same order as the output of the `Pre-processed Datasets` section.)*

**Files inside each folder**:
- `[judge_name]Judge_single_[model].csv` - Single-model evaluation
- `[judge_name]Judge_pair_[model_a]_[model_b].csv` - Pairwise evaluation

**Format**:
```
question_id, category, judge_verdict, judge_explanation, confidence_score (if applicable)
```

**Creation**: Automatically created during Phase 2.1 execution

---

#### Bias Analysis Results

**Language Bias** (`/data/language_bias/`):
- Raw verdict files: `[judge_name]_language_bias_for_pair_llama_deepseek.csv` (config: VN-VN, EN-EN, VN2-VN2)
- Statistics: `/statistics/kappa_result.csv` (Cohen's Kappa scores)

**Position Bias** (`/data/position_bias/`):
- Raw verdict files: `[judge_name]_position_bias_for_pair_llama_deepseek.csv` (config: original, swapped)
- Statistics: `/statistics/bias_result.csv` (Consistency, Biased toward first/second, Error rates)

**Verbosity Bias** (`/data/verbosity_bias/`):
- Raw verdict files: `[judge_name]_verbosity_bias_for_verbosity_llama.csv` (config: original-repetitive, repetitive-original)
- Statistics: `/statistics/bias_by_attempt_and_config_result.csv` (Not Bias, Bias, Error rates)

---

#### Formatted Visualization Output (`/data/visualization/`)

**Language Bias**:
- `language_avg_kappa_ratio_table.csv` - Mean (±Std) Cohen's Kappa scores per judge

**Position Bias**:
- `position_avg_bias_ratio.csv` - Mean (±Std) percentages for Consistency, Biased toward first, Biased toward second, Error

**Verbosity Bias**:
- `verbosity_avg_bias_ratio_table.csv` - Mean (±Std) percentages for Not Bias, Bias, Error

**Agreement**:
- `agreement_ties.csv` - AI vs Human, Human vs Human, Upper limit (with ties)
- `agreement_no_ties.csv` - Same metrics excluding ties
- `agreement_category.csv` - Category-wise agreement breakdown

---

#### Web-based Annotation Dataset (`/data/1500_LLM_responses/`)

**File**: `1500_LLM_responses.csv`

**Composition**: Compiled from 10 pairwise datasets (100 lines each) + 5 single datasets (100 lines each) from `/data/datasets_before_judgement/`

**Purpose**: Curated dataset for the web-based human annotation interface

---

#### Agreement Analysis (`/data/data_for_agreement/`)

**File**: `results.csv`

**Content**: Raw agreement metrics (stringified dictionaries) including:
- AI vs Human agreement ratio
- Human vs Human inter-rater agreement
- Human vs Majority (upper limit)
- Metrics for both tie-included (S1) and tie-excluded (S2) setups
- Category-wise breakdown

### Critical Data Management Notes

- **File Paths**: All notebooks reference paths relative to Google Drive mount point (`/content/drive/MyDrive/ViJudge/`)
- **Path Customization**: Modify notebook path references to match your actual directory structure
- **Incremental Processing**: Notebooks support resuming from checkpoints (e.g., processing questions 80+ if previous run completed questions 0-79)
- **Backup Recommendations**: Create backups before running large-scale evaluations

---

## 4. Complete Execution Pipeline

The evaluation framework consists of four sequential phases. Each phase builds upon outputs from previous phases.

### Phase 1: Collect LLM Responses

**Objective**: Gather responses from multiple LLM models to Vietnamese questions

#### Phase 1.1: `collect_LLMs_responses.ipynb`

**Purpose**: Generate model responses for all questions using Groq or OpenRouter APIs

**Execution Steps**:
1. Open notebook: `phases/1_collect_LLMs_responses/collect_LLMs_responses.ipynb`
2. Mount Google Drive in the first cell
3. Upload `.env` file containing API keys
4. Configure model selection:
   ```python
   groq_models = [
       'openai/gpt-oss-20b',
       'llama-3.3-70b-versatile',
       'moonshotai/kimi-k2-instruct',
       'qwen/qwen3-32b'
   ]
   openrouter_models = ['deepseek/deepseek-chat-v3-0324:free']
   ```
5. Adjust index ranges for incremental processing:
   ```python
   model_question = copy.deepcopy(questions[0:])  # First run: 0:
   model_question = copy.deepcopy(questions[80:]) # Resume: 80:
   ```
6. Select appropriate API and model:
   ```python
   model_answer = gen_model_answer(groq_models[0], groq_client, model_question)
   # model_answer = gen_model_answer(openrouter_models[0], openrouter_client, model_question)
   ```
7. Execute all cells sequentially
8. Output: `drive/MyDrive/ViJudge/data/LLMs_responses/{model_name}_response.csv`

**Key Features**:
- Automatic rate limiting (10-second pause between API calls)
- System prompt in Vietnamese ensuring Vietnamese-language responses
- Multi-turn conversation support
- Error handling with graceful recovery

#### Phase 1.2: `prepare_for_LLM-as-a-Judge.ipynb`

**Purpose**: Preprocess and organize responses for judge evaluation

**Execution Steps**:
1. Open notebook: `phases/1_collect_LLMs_responses/prepare_for_LLM-as-a-Judge.ipynb`
2. Configure the mapping dictionary for the LLMs to be evaluated:
   ```python
   dict_llms = {
       1: ['openai', 'openai_gpt-oss-20b_response.csv'],
       2: ['llama', 'llama-3.3-70b-versatile_response.csv'],
       3: ['moonshotai', 'moonshotai_kimi-k2-instruct_response.csv'],
       4: ['qwen', 'qwen_qwen3-32b_response.csv'],
       5: ['deepseek', 'deepseek_deepseek-chat-v3.1:free_response.csv']
   }
   ```
3. Execute all cells to create datasets. The notebook will automatically iterate and generate both Single and Pairwise datasets.
4. Output: 
   - Single-model datasets: `drive/MyDrive/ViJudge/data/datasets_before_judgement/single_{model_a}.csv`
   - Pairwise datasets: `drive/MyDrive/ViJudge/data/datasets_before_judgement/pair_{model_a}_{model_b}.csv`

**Output Format** (Standardized for both Single and Pairwise datasets):
```text
question_id, category, model_a, model_b, content_turn1, reference_turn1, 
response_turn1_a, response_turn1_b, content_turn2, reference_turn2, 
response_turn2_a, response_turn2_b
```
*(Note: For single-model datasets, `model_b`, `response_turn1_b`, and `response_turn2_b` columns will contain NaN values).*

---

### Phase 2: Evaluation (Automated and Human)

#### Phase 2.1: LLM-as-a-Judge Evaluation

**Objective**: Automated evaluation using LLM models as judges

##### Phase 2.1.1: `judge_for_single_LLM.ipynb`

**Purpose**: Evaluate individual model responses

**Execution Steps**:

1. Open notebook: `phases/2.1_LLM-as-a-Judge/judge_for_single_LLM.ipynb`

2. Mount Google Drive and upload `.env` file

3. Configure judge model:
   ```python
   judge_model = 'openai/gpt-oss-20b'  # Judge model to use
   model_a = 'llama-3.3-70b-versatile' # Model being evaluated
   model_b = ''  # Empty for single-model evaluation
   ```

4. Verify judge prompts are loaded:
   ```python
   judge_prompts_path = 'drive/MyDrive/ViJudge/prompts/judge_prompts.jsonl'
   # Prompts include: single_multi_turn, single_multi_turn_reference, etc.
   ```

5. Set input/output paths (auto-configured):
   ```python
   input_file_path = 'drive/MyDrive/ViJudge/data/datasets_before_judgement/single_llama.csv'
   output_file_path = 'drive/MyDrive/ViJudge/data/gen_judgement/openai/openaiJudge_single_llama.csv'
   ```

6. Execute evaluation:
   - Loads input dataset
   - Iterates through questions
   - Sends to judge API with appropriate prompt
   - Parses verdict from judge response (format: `[[verdict]]`)
   - Saves results incrementally

7. Output: Judge verdicts with explanations

**Supported Verdict Formats**:
- Pairwise: `[[A]]`, `[[B]]`, `[[C]]` (for tie)
- Single: `[[0]]` to `[[10]]` (rating scale)

##### Phase 2.1.2: `judge_for_pair_LLMs.ipynb`

**Purpose**: Pairwise comparison of two models

**Execution Steps** (similar to 2.1.1 with modifications):

1. Configure both models:
   ```python
   judge_model = 'openai/gpt-oss-20b'
   model_a = 'llama-3.3-70b-versatile'
   model_b = 'moonshotai/kimi-k2-instruct'  # Non-empty for pairwise
   ```

2. Input dataset format: pair_[model_a]_[model_b].csv

3. Judge generates pairwise verdicts: A wins, B wins, or Tie

##### Phase 2.1.3: `prepare_for_measure_agreement.ipynb`

**Purpose**: Consolidate scattered individual and pairwise judgment files into aggregated datasets (`[judge_name]_pair.csv` and `[judge_name]_pair_from_single.csv`) to prepare for calculating agreement metrics between judges.

**Execution Steps**:

1. Open notebook: `phases/2.1_LLM-as-a-Judge/prepare_for_measure_agreement.ipynb`

2. Mount Google Drive and define target models and judges:
   ```python
   dict_llms = {1: ['openai'], 2: ['llama'], 3: ['moonshotai'], 4: ['qwen'], 5: ['deepseek']}
   list_judges = ['openai', 'llama', 'moonshotai', 'qwen', 'tinyllama']
   ```

3. Generate pairwise combinations for the target LLMs (e.g., openai vs llama, qwen vs deepseek).

4. **Merge Pairwise Judgments (`[judge_name]_pair.csv`)**:
   - Iterate through the list of judges and generated model combinations.
   - Read individual evaluation files: `[judge_name]Judge_pair_[model_a]_[model_b].csv`
   - Concatenate all pair files for a specific judge into a single DataFrame and export to `[judge_name]_pair.csv`.

5. **Simulate Pairwise Judgments from Single Ratings (`[judge_name]_pair_from_single.csv`)**:
   - Read the corresponding single-model evaluation files for two models: `[judge_name]Judge_single_[model_a].csv` and `[judge_name]Judge_single_[model_b].csv`.
   - Merge the data to align `model_a` and `model_b` conversations.
   - Extract the numeric scores (0-10 rating scale) from the `verdict` column for both models.
   - **Conversion Logic**: Calculate the score difference (`verdict_diff = verdict_a - verdict_b`) to translate independent ratings into pairwise labels:
     - If `verdict_diff > 0`: Label **A** (Model A scored higher)
     - If `verdict_diff < 0`: Label **B** (Model B scored higher)
     - If `verdict_diff == 0`: Label **C** (Tie)
     - Invalid/Missing data: Label **Unknown**
   - Save these derived comparisons into temporary files, then concatenate them all into a final `[judge_name]_pair_from_single.csv` file for each judge.

**Output Files**: 
- `[judge_name]_pair.csv`: Aggregated direct pairwise comparisons.
- `[judge_name]_pair_from_single.csv`: Aggregated pairwise comparisons derived computationally from independent single-model scores.

#### Phase 2.2: Human Evaluation Interface

**Purpose**: Collect human annotations for validation

**Setup Instructions**:

1. Prepare questions file: `data/total_1500_questions/total_1500_questions.csv` (same format as Phase 1)

2. Create and configure a Hugging Face Space:
   - Navigate to Hugging Face and create a new Space.
   - Select **Gradio** as the Space SDK.
   - Go to the Space **Settings** > **Variables and secrets**, and add the following repository secrets:
     ```text
     HF_TOKEN=your_huggingface_token
     DATASET_REPO=your_username/model-evaluation-votes
     ```

3. Deploy the application:
   - Upload the contents of the `phases/2.2_human_judge` directory (including `app.py` and `tex-chtml.js`) to your Space repository.
   - *Note: Ensure your dataset path in `app.py` points correctly to where you store the questions CSV in the Space.*

4. Access the interface:
   - Hugging Face will automatically build and run the Gradio app.
   - Access the evaluation interface directly via your generated Space URL (e.g., `https://huggingface.co/spaces/your_username/your_space_name`).

**Interface Features**:

- **Email Input**: User authentication and result tracking
- **Multi-turn Display**: Shows both question turns with responses
- **Comparison Mode**: Pairwise evaluation (Model A vs Model B)
- **Rating Mode**: Single-model rating (0-10 scale)
- **LaTeX Support**: Renders mathematical expressions via MathJax
- **Cloud Persistence**: Votes automatically saved to Hugging Face Hub

**Evaluation Workflow**:
1. Enter email address
2. View question and model responses
3. Select verdict (A/B/Tie or rating)
4. Provide optional explanation
5. Submit to queue for upload
6. Interface automatically loads next question

**Output Format**:
```
timestamp, question_id, category, model_a, model_b, 
conversation_a, conversation_b, judge_name, judge_type, 
judgement_explanation, verdict, judge_email
```

---

### Phase 3: LLM Judge Bias Analysis

**Objective**: Quantify and visualize systematic biases in LLM-based evaluation

This phase implements three bias analysis experiments:

#### Phase 3.1: Language Bias Analysis

**Execution Pipeline**:
1. **Prepare English version of LLM responses**: Translate original Vietnamese content to English. Locate at `data/datasets_before_judgement/translated/translated_{file_name}.xlsx`.

2. **Generate Verdicts for Language Configurations**: `language_bias_verdict_generation.ipynb`
   - Selects a specific subset of questions for evaluation.
   - Creates three configurations for each evaluation instance: 
     - `VN-VN`: Original Vietnamese responses.
     - `EN-EN`: Translated English responses.
     - `VN2-VN2`: Repeated Vietnamese responses (used as a control baseline).
   - Runs the LLM-as-a-judge multiple times (`n` attempts) across all configurations to collect verdicts.

3. **Statistical Analysis**: `statistics_language_bias.ipynb`
   - Computes **Cohen's Kappa score** to measure the agreement reliability of the judge's verdicts.
   - Compares the baseline (`VN-VN`) against the translated version (`EN-EN`) and the control version (`VN2-VN2`) for each attempt.
   - Output: `data/language_bias/statistics/kappa_result.csv`.

4. **Data Aggregation & Formatting**: `visualization_language_bias.ipynb`
   - Calculates the mean and standard deviation of the Cohen's Kappa scores across all attempts for each judge.
   - Generates a formatted summary table displaying the metrics as `Mean (±Std)`. 

**Output**: A summary table (`data/visualization/language_bias/language_avg_kappa_ratio_table.csv`) containing the average Cohen's Kappa agreement scores and standard deviations.

#### Phase 3.2: Position Bias Analysis

**Execution Pipeline**:

1. **Generate Position-Swapped Verdicts**: `position_bias_verdict_generation.ipynb`
   - Selects a specific subset of questions for evaluation.
   - Creates two configurations for each evaluation instance: 
     - `original`: Preserves the original order of Model A and Model B.
     - `swapped`: Swaps the positions of Model A and Model B (including their respective responses).
   - Runs the LLM-as-a-judge multiple times (`n` attempts) across both configurations to collect verdicts.

2. **Statistical Analysis**: `statistics_position_bias.ipynb`
   - Compares the judgments produced under the original and swapped orderings.
   - Groups the outcomes into four categorical metrics for each attempt:
     - **Consistency**: The judge identifies the same underlying preference (or a tie) across both runs.
     - **Biased toward first**: The judge tends to favor the response appearing in the first position, regardless of the model.
     - **Biased toward second**: The judge tends to favor the response appearing in the second position.
     - **Error**: At least one run does not return a valid label.
   - Output: `data/position_bias/statistics/bias_result.csv`.

3. **Data Aggregation & Formatting**: `visualization_position_bias.ipynb`
   - Calculates the mean and standard deviation of the four metrics (Consistency, Biased toward first, Biased toward second, Error) across all attempts for each judge.
   - Generates a formatted summary table displaying the percentages as `Mean (±Std)`.

**Output**: A formatted summary table (`data/visualization/position_bias/position_avg_bias_ratio.csv`) detailing the Consistency, Biased toward first, Biased toward second, and Error percentages.

#### Phase 3.3: Verbosity Bias Analysis

**Execution Pipeline**:

1. **Prepare repetitive version for responses**: Locate at `data/datasets_before_judgement/verbosity/{file_name}.csv`.

2. **Generate Length-Manipulated Verdicts**: `verbosity_bias_verdict_generation.ipynb`
   - Filters the dataset for evaluation instances that have a generated verbose/repetitive version (`verbosity_response_turn1`).
   - Creates two configurations for each evaluation instance (focusing only on the first turn):
     - `original-repetitive`: Model A provides the original response, Model B provides the verbose/repetitive version.
     - `repetitive-original`: Model A provides the verbose/repetitive version, Model B provides the original response.
   - Runs the LLM-as-a-judge multiple times (`n` attempts) across both configurations to collect verdicts.

3. **Statistical Analysis**: `statistics_verbosity_bias.ipynb`
   - Analyzes the judge's verdicts based on the verbosity configurations to classify the outcomes into three categories:
     - **Not Bias**: The judge correctly selects the original response.
     - **Bias**: The judge favors the repetitive, more verbose response (despite it having no added content quality).
     - **Error**: The judge fails to return a valid choice (A, B, or C).
   - Computes the percentage rates for these metrics per attempt and configuration.
   - Output: `data/verbosity_bias/statistics/bias_by_attempt_and_config_result.csv`.

4. **Data Aggregation & Formatting**: `visualization_verbosity_bias.ipynb`
   - Calculates the mean and standard deviation of the Not Bias, Bias, and Error rates across all attempts, grouped by judge and configuration.
   - Generates a formatted summary table displaying the percentages as `Mean (±Std)`.

**Output**: A formatted summary table (`data/visualization/verbosity_bias/verbosity_avg_bias_ratio_table.csv`) detailing the Not Bias, Bias, and Error percentages, quantifying the judge's inclination toward verbosity.

---

### Phase 4: Measure Agreement

**Objective**: Compare human and automated judge agreements to assess evaluation reliability

#### Phase 4.1: `agreement.ipynb`

**Purpose**: Compute agreement ratios between LLMs and human annotators, as well as human inter-rater reliability.

**Metrics Computed**:
- **AI vs Human**: The ratio of instances where the AI judge's verdict matches the human verdict.
- **Human vs Human**: The ratio of instances where two different human judges agree on the same question.
- **Human vs Majority (Upper Limit)**: The ratio of individual human votes matching the majority human vote.
*(Note: All metrics are calculated for two setups: S1 (including ties) and S2 (excluding ties), along with a category-wise breakdown).*

**Execution Steps**:
1. Load human judge datasets (`human_pair_with_inconsistencies_cleaned.csv`).
2. Load LLM judge verdicts for both evaluation types (`_pair.csv` and `_pair_from_single.csv`).
3. Iterate through target models (e.g., openai, moonshotai, llama, qwen, tinyllama).
4. Compute the agreement ratios across all overlapping question/model-pair keys.
5. Output: `data/data_for_agreement/results.csv` (contains stringified dictionaries for Ties, No_ties, and Category metrics).

#### Phase 4.2: `visualization_agreement.ipynb`

**Purpose**: Parse, format, and structure the nested agreement results into clean tabular data for reporting.

**Execution Steps**:
- Reads the raw `results.csv`.
- Parses the `Ties` column into distinct metrics (`AI_vs_Human`, `Human_vs_Human`, `Upper_limit`).
- Parses the `No_ties` column into identical distinct metrics for the tie-excluded setup.
- Expands the `Category` column into a detailed breakdown, formatting the values as `ratio (agreed/total)`.

**Output Tables**:
- Formatted DataFrames for setup S1 (Ties) and S2 (No Ties).
- `agreement_ties.csv`: Agreement ratios with **ties included** (Setup S1)
- `agreement_no_ties.csv`: Agreement ratios with **ties excluded** (Setup S2)
- `agreement_category.csv`: Category-specific agreement breakdown for each model and evaluation type

---