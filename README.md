# LLM Bias and Safety Evaluation Framework

A comprehensive research framework for evaluating bias, toxicity, and safety risks in Large Language Models across healthcare, workplace, and smart city scenarios, featuring advanced analysis, visualization, and reporting capabilities.

## Project Overview

This framework provides end-to-end evaluation of LLM bias and safety through:

- **Baseline Bias Assessment**: Quantitative measurement of age, gender, and racial bias
- **Advanced Mitigation Strategies**: Two distinct approaches to bias reduction
- **Comprehensive Analysis**: Statistical analysis with publication-ready visualizations
- **Multi-Model Support**: Evaluation across 4 state-of-the-art LLMs
- **Domain-Specific Testing**: Healthcare, workplace, and smart city scenarios

## Key Features

### Bias Detection Capabilities
- **Multi-dimensional Analysis**: Age, gender, and racial bias detection
- **Context-Aware Evaluation**: Domain-specific bias assessment
- **Statistical Significance**: Rigorous statistical analysis and reporting
- **Comparative Analysis**: Cross-model and cross-strategy comparisons

### Supported LLM Models
- **Microsoft Phi-3.5**: `microsoft/Phi-3.5-mini-instruct`
- **Mistral-7B**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Qwen2.5-7B**: `Qwen/Qwen2.5-7B-Instruct`
- **Falcon-7B**: `tiiuae/falcon-7b-instruct`

### Mitigation Strategies
1. **Prompt Engineering (Strategy 1)**
   - Ethical guardrails and bias-minimizing prompts
   - Bias-aware response generation

2. **LLM-as-Judge (Strategy 2)**
   - Four-phase evaluation pipeline
   - Automated bias detection and repair
   - Quality-preserving mitigation

### Analysis & Visualization
- **Interactive Dashboards**: Comprehensive bias analysis charts
- **Heatmaps**: Model-strategy performance matrices
- **Statistical Reports**: Publication-ready analysis
- **Excel Exports**: Detailed scenario-by-scenario results

## Project Structure

```
LLM Bias and Safety Evaluation/
├── src/                                    # Core implementation
│   ├── baseline_results.ipynb                 # Baseline bias assessment (Jupyter)
│   ├── mitigation strategy 1.ipynb            # Prompt engineering mitigation (Jupyter)
│   ├── mitigation strategy 2.ipynb            # LLM-as-Judge mitigation (Jupyter)
│   ├── bias_analyzer_graph.py                 # Comprehensive 9-graph analysis
│   ├── mitigation_bias_analyzer.py            # Mitigation analysis
│   └── requirements.txt                       # Dependencies
│
├── Data/                                   # Scenarios and test data
│   ├── sample_scenarios.jsonl                 # Core ethical scenarios
│
├── LLMs Responses/                         # Baseline model outputs
│   ├── baseline_phi35.csv
│   ├── baseline_mistral-7b.csv
│   ├── baseline_qwen2.5-7b.csv
│   └── baseline_falcon-7b.csv
│
├── LLM mitigated responses/                # Mitigation results
│   ├── mitig_prompt_aligned_norefuse_*.csv     # Strategy 1 results
│   └── mitig_judge_*.csv                       # Strategy 2 results
│
├── Comprehensive Analysis Graphs/          # Complete 9-graph analysis output
├── Comparison Graphs/                      # Baseline analysis visualizations
├── Mitigation Analysis Graphs/             # Mitigation analysis charts
└── Final Report pdf/                       # Research documentation
```

## Getting Started

### Prerequisites

- **Google Colab Pro** (recommended for GPU access)
- **Python 3.8+**
- **CUDA-capable GPU** (T4 or better)
- **Hugging Face Account** (for model access)

### Installation & Setup

#### 1. Environment Setup (Google Colab)

```python
# Install required packages (run in Colab)
!pip -q install bitsandbytes==0.43.3
!pip -q install transformers==4.43.3 accelerate==0.33.0
!pip -q install sentence-transformers==2.2.2 detoxify==0.5.2
!pip -q install pandas tqdm matplotlib seaborn

# Set CUDA memory optimization
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

#### 2. Hugging Face Authentication

```python
from huggingface_hub import login
login()  # Enter your HF token when prompted
```

#### 3. Local Analysis Setup (Cursor/VS Code)

```bash
# Install dependencies
pip install -r src/requirements.txt

# Set up project directory
cd "Main LLM SAFETY"
```

## Usage Guide

### Phase 1: Baseline Assessment

#### Step 1: Configure Model and Run Baseline
```python
# Edit configuration in baseline_results.ipynb (Google Colab)
MODEL_ID = "microsoft/Phi-3.5-mini-instruct"  # Choose your model
MODEL_KEY = "phi35"
MAX_NEW_TOKENS = 120

# Run baseline assessment notebook
# Execute all cells in baseline_results.ipynb
```

#### Step 2: Generate Comprehensive Analysis
```python
# Run comprehensive 9-graph analysis
python src/bias_analyzer_graph.py
```

**Generated Outputs (9 Comprehensive Graphs):**
- Age bias analysis charts
- Racial bias distribution
- Gender bias patterns
- Score comparisons (similarity and toxicity)
- Overall model rankings
- Healthcare detailed analysis
- Workplace detailed analysis
- Moral dilemma analysis
- Comprehensive scenario ranking

### Phase 2: Mitigation Implementation

#### Strategy 1: Prompt Engineering
```python
# Configure and run prompt engineering mitigation in Google Colab
TARGET_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
MODEL_KEY = "phi35"

# Execute mitigation notebook
# Execute all cells in "mitigation strategy 1.ipynb"
```

**Key Features:**
- Ethical guardrails integration
- Zero-refusal prompting
- Bias-minimizing response selection
- Quality-preserving techniques

#### Strategy 2: LLM-as-Judge
```python
# Configure and run LLM-as-Judge mitigation in Google Colab
TARGET_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
MODEL_KEY = "phi35"

# Execute mitigation notebook
# Execute all cells in "mitigation strategy 2.ipynb"
```

**Pipeline Phases:**
1. Initial response generation
2. Bias detection (using judge model)
3. Response evaluation
4. Automated bias repair

### Phase 3: Comprehensive Analysis

#### Complete 9-Graph Analysis
```python
# Run comprehensive analysis for all scenarios and models
python src/bias_analyzer_graph.py
```

**Generated Analysis:**
1. Age Bias Analysis (young vs old preferences)
2. Racial Bias Analysis (racial preference patterns)
3. Gender Bias Analysis (gender preference patterns)
4. Score Comparisons (similarity and toxicity)
5. Overall Model Ranking (comprehensive performance)
6. Healthcare Detailed Analysis (medical scenario bias)
7. Workplace Detailed Analysis (employment bias patterns)
8. Moral Dilemma Analysis (ethical decision-making)
9. Comprehensive Scenario Ranking (all scenarios ranked)

#### Mitigation-Specific Analysis
```python
# Run mitigation strategy analysis
python src/mitigation_bias_analyzer.py
```

## Analysis Outputs

### Comprehensive 9-Graph Analysis Results
- **Graph 1**: Age Bias Analysis (4 subplots: model comparison, domain comparison, heatmap, ranking)
- **Graph 2**: Racial Bias Analysis (4 subplots: model comparison, domain comparison, heatmap, ranking)
- **Graph 3**: Gender Bias Analysis (4 subplots: model comparison, domain comparison, heatmap, ranking)
- **Graph 4**: Score Comparisons (4 subplots: similarity by domain, toxicity by domain, similarity ranking, toxicity ranking)
- **Graph 5**: Overall Model Ranking (4 subplots: bias breakdown, similarity vs toxicity, bias ranking, summary table)
- **Graph 6**: Healthcare Detailed Analysis (6 subplots: bias by scenario, similarity scores, problematic scenarios, age/race bias, model performance)
- **Graph 7**: Workplace Detailed Analysis (6 subplots: bias by scenario, similarity scores, problematic scenarios, gender/age bias, model performance)
- **Graph 8**: Moral Dilemma Analysis (4 subplots: bias by model, similarity by model, age preferences)
- **Graph 9**: Comprehensive Scenario Ranking (4 subplots: scenario ranking, similarity ranking, heatmap, overall performance)

### Mitigation Analysis Results
- **Strategy Effectiveness**: Before/after bias reduction comparison
- **Cross-Model Comparison**: Performance across all 4 models
- **Domain-Specific Analysis**: Healthcare, workplace, smart city breakdowns
- **Scenario Breakdown**: Detailed per-scenario analysis

### Advanced Analysis Features
- **Statistical Significance Testing**
- **Multi-dimensional Bias Analysis**
- **Pattern Recognition across Domains**
- **Publication-Ready Visualizations**

## Configuration Options

### Model Configuration
```python
# Available models and their configurations
MODELS = {
    "phi35": "microsoft/Phi-3.5-mini-instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3", 
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "falcon-7b": "tiiuae/falcon-7b-instruct"
}
```

### Scenario Configuration
```python
# Scenario domains
DOMAINS = {
    'healthcare': 'Medical decisions and resource allocation',
    'workplace': 'Hiring, promotion, and workplace decisions',
    'smart_city': 'Urban planning and public service allocation'
}
```

### Analysis Configuration
```python
# Output directories (customize as needed)
OUTPUT_DIRS = {
    'comprehensive': "Comprehensive Analysis Graphs/",
    'baseline': "Comparison Graphs/",
    'mitigation': "Mitigation Analysis Graphs/"
}

# Comprehensive analysis generates 9 graphs:
GRAPH_OUTPUTS = [
    'age_bias_analysis.png',
    'racial_bias_analysis.png', 
    'gender_bias_analysis.png',
    'score_comparisons.png',
    'overall_model_ranking.png',
    'healthcare_detailed_analysis.png',
    'workplace_detailed_analysis.png',
    'moral_dilemma_analysis.png',
    'comprehensive_scenario_ranking.png'
]
```

## Key Metrics

### Bias Metrics
- **Age Bias**: Younger vs older preference percentages
- **Racial Bias**: White vs other race preference percentages  
- **Gender Bias**: Male vs female preference percentages
- **Overall Bias**: Combined bias detection rate

### Quality Metrics
- **Similarity Score**: Cosine similarity to expected responses (0-1)
- **Toxicity Score**: Detoxify-based toxicity measurement (0-1)
- **Bias Severity**: Combined severity scoring (0-4)

### Effectiveness Metrics
- **Bias Reduction**: Percentage point reduction from baseline
- **Quality Preservation**: Similarity score maintenance
- **Strategy Comparison**: Relative effectiveness analysis

## Technical Implementation

### Core Technologies
- **PyTorch**: Model loading and inference
- **Transformers**: Hugging Face model integration
- **BitsAndBytesConfig**: 4-bit quantization for memory efficiency
- **Sentence Transformers**: Semantic similarity evaluation
- **Detoxify**: Toxicity detection
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Visualization and charting

### Memory Optimization
- **4-bit Quantization**: Reduces VRAM usage by ~75%
- **CPU Offloading**: Evaluation models kept on CPU
- **Garbage Collection**: Automatic memory cleanup
- **Batch Processing**: Efficient scenario processing

### Quality Assurance
- **Deterministic Generation**: `temperature=0.0, do_sample=False`
- **Robust Parsing**: Multiple fallback parsing strategies
- **Error Handling**: Graceful failure management
- **Data Validation**: Comprehensive input validation

## Research Applications

### Academic Research
- **Bias Detection Studies**: Quantitative bias measurement
- **Fairness Evaluation**: Comparative mitigation analysis  
- **Safety Assessment**: Toxicity and harm evaluation
- **Model Benchmarking**: Systematic model comparison

### Industry Applications
- **AI Safety Auditing**: Comprehensive model evaluation
- **Bias Mitigation**: Practical bias reduction implementation
- **Compliance Assessment**: Regulatory requirement evaluation
- **Risk Management**: Safety risk identification

## Example Workflow

### Complete Evaluation Pipeline
```bash
# 1. Run baseline for all models (Google Colab Pro)
# Execute baseline_results.ipynb for each model:
# - Configure MODEL_ID and MODEL_KEY
# - Run all cells for phi35, mistral-7b, qwen2.5-7b, falcon-7b

# 2. Run mitigation strategies (Google Colab Pro)
# Execute mitigation strategy 1.ipynb for each model
# Execute mitigation strategy 2.ipynb for each model

# 3. Generate comprehensive analysis (Local/Cursor)
python src/bias_analyzer_graph.py             # Complete 9-graph analysis
python src/mitigation_bias_analyzer.py         # Mitigation analysis
```

## Troubleshooting

### Common Issues

#### Memory Issues
```python
# Reduce model size or batch size
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"  # Use smaller model
MAX_NEW_TOKENS = 80  # Reduce generation length
```

#### CUDA Errors
```python
# Enable memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()  # Clear CUDA cache
```

#### Authentication Issues
```python
# Set Hugging Face token
os.environ["HF_TOKEN"] = "your_hf_token_here"
```

### File Path Issues
- Ensure all CSV files are in correct directories
- Verify file naming conventions match the code
- Check file permissions and accessibility

## Expected Results

### Baseline Results
- **Bias Detection**: 35-45% overall bias rate across models
- **Similarity Scores**: 0.55-0.65 average similarity
- **Toxicity Levels**: <0.05 average toxicity

### Mitigation Results  
- **Bias Reduction**: 60-80% reduction from baseline
- **Quality Preservation**: >90% similarity retention
- **Strategy Effectiveness**: LLM-as-Judge typically outperforms Prompt Engineering

## Advanced Features

### Custom Scenario Integration
```python
# Add custom scenarios to sample_scenarios.jsonl
{
    "id": "custom_001",
    "domain": "healthcare", 
    "prompt": "Your custom scenario...",
    "expected_response": "Expected ethical response...",
    "protected_attribute": "age"
}
```

### Extended Analysis
- **Temporal Analysis**: Bias trends over time
- **Intersectional Analysis**: Multiple protected attributes
- **Domain-Specific Deep Dives**: Focused analysis by domain

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{llm_bias_safety_evaluation,
  title={Evaluating and Mitigating Bias and Safety in Smart-City and Workplace domains for LLMs},
  author={[Aryan Anchan]},
  year={2025},
  description={Comprehensive framework for evaluating and mitigating bias in Large Language Models},
  url={[Your Repository URL]}
}
```

## Contributing

We welcome contributions! Please see our contribution guidelines for:
- Code style requirements
- Testing procedures  
- Documentation standards
- Issue reporting

## License

This project is licensed under the MIT License - see the LICENSE file for details.
