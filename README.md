# LLM Bias and Safety Evaluation Framework

This is a README file for baseline assessment and two advanced mitigation strategies, with extensive visualization and analysis capabilities.

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
│   ├── baseline_results.py                    # Baseline bias assessment
│   ├── mitigation strategy 1.py               # Prompt engineering mitigation
│   ├── mitigation strategy 2.py               # LLM-as-Judge mitigation
│   ├── improved_bias_analyzer.py              # Baseline analysis & visualization
│   ├── mitigation_bias_analyzer.py            # Mitigation analysis
│   ├── enhanced_mitigation_bias_analyzer.py   # Advanced mitigation analysis
│   ├── bias_pattern_analyzer.py               # Pattern recognition analysis
│   └── requirements.txt                       # Dependencies
│
├── Data/                                   # Scenarios and test data
│   ├── sample_scenarios.jsonl                 # Core ethical scenarios
│   ├── ethical_scenarios.jsonl                # Additional test cases
│   └── test_texts.csv                         # Sample evaluation data
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
├── Comparison Graphs/                      # Baseline analysis visualizations
├── Mitigation Analysis Graphs/             # Mitigation analysis charts
├── Enhanced Mitigation Analysis Graphs/    # Advanced analysis results
├── Detailed Scenario Analysis/             # Scenario-specific analysis
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
# Edit configuration in baseline_results.py
MODEL_ID = "microsoft/Phi-3.5-mini-instruct"  # Choose your model
MODEL_KEY = "phi35"
MAX_NEW_TOKENS = 120

# Run baseline assessment
python src/baseline_results.py
```

#### Step 2: Generate Baseline Analysis
```python
# Run comprehensive baseline analysis
python src/improved_bias_analyzer.py
```

**Generated Outputs:**
- Age bias analysis charts
- Racial bias distribution
- Gender bias patterns
- Similarity score evaluations
- Overall model rankings

### Phase 2: Mitigation Implementation

#### Strategy 1: Prompt Engineering
```python
# Configure and run prompt engineering mitigation
TARGET_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
MODEL_KEY = "phi35"

# Execute mitigation
python "src/mitigation strategy 1.py"
```

**Key Features:**
- Ethical guardrails integration
- Zero-refusal prompting
- Bias-minimizing response selection
- Quality-preserving techniques

#### Strategy 2: LLM-as-Judge
```python
# Configure and run LLM-as-Judge mitigation
TARGET_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
MODEL_KEY = "phi35"

# Execute mitigation
python "src/mitigation strategy 2.py"
```

**Pipeline Phases:**
1. Initial response generation
2. Bias detection (using judge model)
3. Response evaluation
4. Automated bias repair

### Phase 3: Comprehensive Analysis

#### Basic Mitigation Analysis
```python
# Run standard mitigation analysis
python src/mitigation_bias_analyzer.py
```

#### Advanced Analysis
```python
# Run enhanced mitigation analysis
python src/enhanced_mitigation_bias_analyzer.py
```

#### Pattern Recognition Analysis
```python
# Run deep pattern analysis
python src/bias_pattern_analyzer.py
```

## Analysis Outputs

### Baseline Analysis Results
- **Age Bias Charts**: Young vs old preference patterns
- **Racial Bias Heatmaps**: White vs other race preferences
- **Gender Bias Analysis**: Male vs female selection patterns
- **Similarity Scores**: Response quality measurements
- **Model Rankings**: Comprehensive performance comparison

### Mitigation Analysis Results
- **Strategy Effectiveness**: Before/after bias reduction
- **Cross-Model Comparison**: Performance across all models
- **Domain-Specific Analysis**: Healthcare, workplace, smart city
- **Scenario Breakdown**: Detailed per-scenario analysis
- **Excel Exports**: Comprehensive data tables

### Advanced Analysis Features
- **Statistical Significance Testing**
- **Correlation Analysis**
- **Pattern Recognition**
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
    'baseline': "Comparison Graphs/",
    'mitigation': "Mitigation Analysis Graphs/", 
    'enhanced': "Enhanced Mitigation Analysis Graphs/",
    'detailed': "Detailed Scenario Analysis/"
}
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
# 1. Run baseline for all models
for model in phi35 mistral-7b qwen2.5-7b falcon-7b; do
    # Configure MODEL_ID and MODEL_KEY in baseline_results.py
    python src/baseline_results.py
done

# 2. Run mitigation strategies
for model in phi35 mistral-7b qwen2.5-7b falcon-7b; do
    # Configure model in mitigation strategy files
    python "src/mitigation strategy 1.py"
    python "src/mitigation strategy 2.py"  
done

# 3. Generate comprehensive analysis
python src/improved_bias_analyzer.py           # Baseline analysis
python src/mitigation_bias_analyzer.py         # Mitigation analysis  
python src/enhanced_mitigation_bias_analyzer.py # Advanced analysis
python src/bias_pattern_analyzer.py            # Pattern analysis
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
