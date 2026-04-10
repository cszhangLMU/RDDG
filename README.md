# Self-Reinforcing Controllable Synthesis of Rare Relational Data via Bayesian Calibration

## Abstract

Imbalanced data commonly present in real-world applications, making imbalanced learning a long-standing research concern. While data synthesis offers an effective solution to mitigating rare-class data scarcity and LLMs have revolutionized text generation, the  application of LLMs to relational/structured tabular  data synthesis remains under-explored. Moreover, existing methods lack an effective feedback mechanism that can guide LLMs towards continuously optimizing the quality of the generated data throughout the in-context learning process. To address these gaps, we propose RDDG, Relational Data generator with Dynamic Guidance, which is a unified in-context learning framework that employs progressive chain-of-thought (CoT) steps to generate tabular data to enhance downstream imbalanced  classification performance. RDDG first uses core set selection to identify representative samples from the original data, next utilizes in-context learning to discover the inherent patterns and correlations among attributes in the core set, then generates tabular data while preserving the above constraints. More importantly, it devises a self-reinforcing mechanism that provides automatic feedback on the quality of the generated data, enabling continuous quality optimization throughout the generation process. Experimental results on multiple real and synthetic datasets demonstrate that RDDG significantly outperforms existing approaches in both data fidelity and downstream imbalanced classification performance.  We make our code available at [here](https://github.com/colored32/RDDG).

<div style="text-align: center;">
    <img src="img/main diagram 3_01.png" width="600" height="300" />
</div>

## Installation and Environment Setup

* Clone the repository

```shell
git clone https://github.com/colored32/RDDG.git

# create environment
conda create -n llm python=3.9 -y
conda activate llm
pip install -r requirements.txt
```

## Quick start

### Usage

1. **Configure OpenAI API Key**: Enter your OpenAI API key in `codes/SyntheticDataGeneration/generate_samples_Sick.py`:

   ```python
   (line 36) openai_key = "Your-OpenAI-Key"
   ```
2. **Generate Synthetic Datasets**:

   To generate synthetic datasets using our method, run the following command:

   ```bash
   cd codes/SyntheticDataGeneration
   python generate_samples_Sick.py
   cd ..
   ```
   If you want use other dataset:

   ```bash
   python generate_samples_Thyroid.py 
   ```
3. **Train and Evaluate Downstream Task Models**:
   To evaluate the quality of the synthetic data, use the following command:

   ```bash
   cd ../DownstreamTasks
   python Classification.py  
   ```
4. **Change Model to generate Synthetic Datasets**

   ```bash
   <!-- Use Llama3 Model -->
   cd ./codes/SyntheticDataGeneration/otherLLMs/llama
   python generate_samples_Sick_llama.py

   <!-- Use Mistral Model-->
   cd ./codes/SyntheticDataGeneration/otherLLMs/mistral
   python generate_samples_Sick_mistral.py 
   ```
