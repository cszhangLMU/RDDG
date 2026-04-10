import openai
import os
from dotenv import load_dotenv
import pandas as pd
import string
import random
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from util import get_prompt_conclass, parse_prompt2df, parse_result, get_unique_features, make_final_prompt, \
    weight_init, compute_eva_score, generate_prompt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import MLP
import numpy as np
import re

# Configure proxy if needed
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

# OpenRouter API configuration
api_key = "YOUR-API-KEY"  # Replace with your OpenRouter API key
base_url = "https://openrouter.ai/api/v1"

params = {
    "openai_key": api_key,  # Use OpenRouter API key
    "base_url": base_url,
    "model": "meta-llama/llama-3.1-8b-instruct",  # Specify LLaMA model from OpenRouter
    "DATA_NAME": "Sick",
    "TARGET": "Class",
    "N_CLASS": 2,
    "N_SAMPLES_PER_CLASS": 15,
    "N_SET": 4,
    "USE_RANDOM_WORD": True,
    "N_CORESETS_BATCH": 1,
    "N_BATCH": 20,
    "MODEL_NAME": "Sick_STPrompt_LLama",
    "N_TARGET_SAMPLES": 1000,
}

# Configure parameters
config = {
    "early_window": (0, 10),
    "late_window": (90, 100),
    "eva_epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.001,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "seed": 0  # Set random seed
}
params.update({
    "DATA_DIR": f"../../../../data/realdata/{params['DATA_NAME']}",
    "SAVE_DIR": f"../../../../data/syndata/{params['MODEL_NAME']}"
})

# Initialize OpenRouter API
load_dotenv()
openai.api_key = params['openai_key']
openai.api_base = params['base_url']
os.environ["OPENAI_API_KEY"] = params['openai_key']

# Initialize LangChain with OpenRouter's LLaMA model
llm = ChatOpenAI(
    model=params["model"],
    openai_api_key=params["openai_key"],
    openai_api_base=params["base_url"],
    temperature=0.0
)
output_parser = StrOutputParser()

# Initialize parameters
DATA_NAME = params['DATA_NAME']
TARGET = params['TARGET']
REAL_DATA_SAVE_DIR = params['DATA_DIR']
symModel = params['MODEL_NAME']
SYN_DATA_SAVE_DIR = params['SAVE_DIR']
os.makedirs(SYN_DATA_SAVE_DIR, exist_ok=True)

# Read real data
X_train = pd.read_csv(os.path.join(REAL_DATA_SAVE_DIR, f'X_train_1.csv'), index_col=None)
y_train = pd.read_csv(os.path.join(REAL_DATA_SAVE_DIR, f'y_train_1.csv'), index_col=None)
data = pd.concat((y_train, X_train), axis=1)
data_copy = data.copy()

# Sick dataset categorical features
CATEGORICAL_FEATURES = ['sex', 'on_thyroxine', 'query_on_thyroxine',
                        'on_antithyroid_medication', 'sick', 'pregnant',
                        'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
                        'query_hyperthyroid', 'lithium', 'goitre', 'tumor',
                        'hypopituitary', 'psych', 'TSH_measured', 'T3_measured',
                        'TT4_measured', 'T4U_measured', 'FTI_measured',
                        'referral_source', 'Class']

encoder = LabelEncoder()
for feature in CATEGORICAL_FEATURES:
    data_copy[feature] = encoder.fit_transform(data_copy[feature])

X = data_copy.drop('Class', axis=1)
y_train_encoded = data_copy['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train_scaled_tensor = torch.tensor(X_train_scaled.to_numpy(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded.to_numpy(), dtype=torch.long)

input_dim = X_train_scaled.shape[1]
hidden_dim = 64
output_dim = len(y_train_tensor.unique())

criterion = nn.CrossEntropyLoss()

train_data = TensorDataset(X_train_scaled_tensor, y_train_tensor)
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

model = MLP(input_dim, output_dim)
model.apply(weight_init)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.9))

data_name = "sick"
device = config["device"]

dataset_variance_total = '../../dataset_variance_total/' + str(data_name) + '_variance_total.npy'
if not os.path.exists(dataset_variance_total):
    variance_total = compute_eva_score(train_loader, model, config["eva_epochs"], config["early_window"],
                                       config["late_window"], optimizer, criterion, device=device)
    np.save(dataset_variance_total, variance_total)

variance_total = np.load(dataset_variance_total)

coreset_path = os.path.join('../../coreset', f'{data_name}_selection.csv')

if not os.path.exists(coreset_path):
    num_selected_samples_per_class = 100
    selected_indices = []

    for class_idx in range(output_dim):
        class_indices = np.where(y_train_encoded == class_idx)[0]
        class_scores = variance_total[class_indices]
        top_indices = class_indices[np.argsort(-class_scores)[:num_selected_samples_per_class]]
        selected_indices.extend(top_indices)

    data_coreset = data.iloc[selected_indices]
    data_coreset.to_csv(coreset_path, index=None)
    print(f'data_coreset saved to {coreset_path}')

data_coreset = pd.read_csv(coreset_path)

NAME_COLS = ','.join(data.columns) + '\n'
unique_categorical_features = get_unique_features(data, CATEGORICAL_FEATURES)
unique_categorical_features['Class'] = ['sick', 'negative']
cat_idx = []
for i, c in enumerate(X_train.columns):
    if c in CATEGORICAL_FEATURES:
        cat_idx.append(i)

N_CLASS = params['N_CLASS']
N_SAMPLES_PER_CLASS = params['N_SAMPLES_PER_CLASS']
N_SET = params['N_SET']
N_CORESETS_BATCH = params['N_CORESETS_BATCH']
N_BATCH = params['N_BATCH']
N_CORESETS_TOTAL = N_SAMPLES_PER_CLASS * N_SET * N_CORESETS_BATCH
N_SAMPLES_TOTAL = N_SAMPLES_PER_CLASS * N_SET * N_BATCH
mapper_r = {}

# Apply random word strategy
if params['USE_RANDOM_WORD']:
    def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
        first = ''.join(random.choice(string.ascii_uppercase) for _ in range(1))
        left = ''.join(random.choice(chars) for _ in range(size - 1))
        return first + left

    def make_random_categorical_values(unique_categorical_features):
        mapper = {}
        mapper_r = {}
        new_unique_categorical_features = {}
        for c in unique_categorical_features:
            mapper[c] = {}
            mapper_r[c] = {}
            new_unique_categorical_features[c] = []

            for v in unique_categorical_features[c]:
                a = id_generator(3)
                new_unique_categorical_features[c].append(a)
                mapper[c][v] = a
                mapper_r[c][a] = v
        return mapper, mapper_r, new_unique_categorical_features

    mapper, mapper_r, unique_categorical_features = make_random_categorical_values(unique_categorical_features)

    for c in mapper:
        data_coreset[c] = data_coreset[c].map(lambda x: mapper[c][x])
    for c in mapper:
        data[c] = data[c].map(lambda x: mapper[c][x])

# Make prompt template
initial_prompt = """Class: hypothyroidism is a condition in which the thyroid gland is underperforming or producing too little thyroid hormone,
age: the age of an patient,
sex: the biological sex of an patient,
TSH: thyroid stimulating hormone,
T3: triiodothyronine hormone,
TT4: total levothyroxine hormone,
T4U: levothyroxine hormone uptake,
FTI: free levothyroxine hormone index,
referral_source: institution that supplied the thyroid disease record.\n\n
"""

numbering = ['A', 'B', 'C', 'D']
prompt = get_prompt_conclass(initial_prompt, numbering, N_SAMPLES_PER_CLASS, N_CLASS, N_SET, NAME_COLS)

# Phase 1: Analyze variable relationships
def analyze_variable_relationships(data_coreset, prompt, llm, unique_categorical_features, output_parser, TARGET,
                                   N_CORESETS_TOTAL, N_CORESETS_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, N_CLASS):
    relationship_analysis_prompt = """Please analyze the relationships between the provided attributes. Using common knowledge, employ various statistical methods (e.g., correlation analysis) to explore how patient age, sex, thyroid hormone levels (TSH, T3, TT4, T4U, FTI), and medical history factors (e.g., thyroid medication, surgery, goitre) 
    relate to the class variable (A or B). Identify any significant correlations or patterns that could provide insights into the diagnosis of hypothyroidism or reveal potential interactions between these attributes. Remember, I only 
    need you to provide the analysis results, not the methods for calculation.\n\n"""

    prompt1 = prompt + relationship_analysis_prompt
    prompt_template1 = PromptTemplate.from_template(prompt1)
    llm1 = (
            prompt_template1
            | llm
            | output_parser
    )
    final_prompt1, inputs_batch = make_final_prompt(unique_categorical_features, TARGET, data_coreset, prompt_template1,
                                                    N_CORESETS_TOTAL, N_CORESETS_BATCH, N_SAMPLES_PER_CLASS, N_SET,
                                                    NAME_COLS, N_CLASS)
    analysis_results = llm1.batch(inputs_batch)
    return analysis_results[0]

analysis_result = analyze_variable_relationships(data_coreset, prompt, llm, unique_categorical_features, output_parser,
                                                 TARGET, N_CORESETS_TOTAL, N_CORESETS_BATCH, N_SAMPLES_PER_CLASS, N_SET,
                                                 NAME_COLS, N_CLASS)

# Phase 2: Define data generation constraints
def define_generation_constraints(analysis_results, initial_prompt, llm):
    constraints_prompt = f"""
    {analysis_results}\n
    Based on the above background data, the data, and the relationships between the data, establish rules and 
    constraints for data generation.
    """
    prompt2 = initial_prompt + constraints_prompt
    constraints = llm.invoke(prompt2).content
    return constraints

constraints = define_generation_constraints(analysis_result, initial_prompt, llm)

def fix_and_clean(s: str, name_cols: str, classA: str, classB: str, expected_cols: int) -> str:
    lines = [ln for ln in s.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        t = ln.strip()
        if t == name_cols.strip():
            continue
        if t in {"A.", "B."}:
            continue
        if t.startswith("A.,"):
            t = t.replace("A.,", f"{classA},", 1)
        elif t.startswith("B.,"):
            t = t.replace("B.,", f"{classB},", 1)
        if t.count(",") == expected_cols - 1:
            out.append(t)
    return "\n".join(out) + ("\n" if out else "")

# Phase 3: Generate data
def generate_data(constraints, feedback, data, llm, unique_categorical_features, output_parser, TARGET, N_SAMPLES_TOTAL,
                  N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, N_CLASS, input_df_all, synthetic_df_all):
  
    class_tokens = unique_categorical_features['Class']
    classA, classB = class_tokens[0], class_tokens[1]
    
    extra_rules = f"""
Sections:
- 'A.' means Class == '{classA}'
- 'B.' means Class == '{classB}'
For every data row:
- The FIRST column MUST be the class token ('{classA}' or '{classB}') according to the section.
- Never output 'A.' or 'B.' inside any CSV row.
- 'A.' and 'B.' are only section headers. Data rows must start directly with the class token.
"""
    
    generation_prompt = f"""
    {constraints}.{extra_rules}
    Generate synthetic data that follows the same patterns as the provided examples. 
Return ONLY the data in CSV format, no explanations or code.
Ensure balanced class generation.

Format: Return data exactly like the examples below, with comma-separated values.
    """
    prompt = get_prompt_conclass("", numbering, N_SAMPLES_PER_CLASS, N_CLASS, N_SET, NAME_COLS)
    prompt += NAME_COLS
    

    text_results = []
    columns1 = data.columns
    columns2 = list(data.columns)
    err = []
    while len(synthetic_df_all) < params['N_TARGET_SAMPLES']:
      
        prompt3 = generation_prompt + feedback + prompt
        prompt3_escaped = prompt3.replace('{', '{{').replace('}', '}}')

        import re
        prompt3 = re.sub(r'\{\{(v\d+)\}\}', r'{\1}', prompt3_escaped)
        prompt_template3 = PromptTemplate.from_template(prompt3)
        llm3 = (prompt_template3
                | llm
                | output_parser)
      
        final_prompt3, inputs_batch = make_final_prompt(unique_categorical_features, TARGET, data, prompt_template3,
                                                        N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS,
                                                        N_CLASS)
        
        
        inter_text = llm3.batch(inputs_batch)
        for i in range(len(inter_text)):
            try:
                text_results.append(final_prompt3[i].text + inter_text[i])
                input_df = parse_prompt2df(final_prompt3[i].text, split=NAME_COLS, inital_prompt=initial_prompt,
                                           col_name=columns1)
                text = fix_and_clean(inter_text[i], NAME_COLS, classA, classB, len(data.columns))
                result_df = parse_result(text, NAME_COLS, columns2, CATEGORICAL_FEATURES,
                                         unique_categorical_features)

                input_df_all = pd.concat([input_df_all, input_df], axis=0)
                synthetic_df_all = pd.concat([synthetic_df_all, result_df], axis=0)
            except Exception as e:
                err.append(inter_text[i])
        print('Number of Generated Samples:', len(synthetic_df_all), '/', params['N_TARGET_SAMPLES'])
        feedback = quality_evaluation(synthetic_df_all, data)
    return synthetic_df_all
  
def quality_evaluation(generated_data, real_data):
    evaluation_results = {}
    
    numeric_cols_gen = generated_data.select_dtypes(include=[np.number]).columns
    numeric_cols_real = real_data.select_dtypes(include=[np.number]).columns
    common_numeric_cols = list(set(numeric_cols_gen) & set(numeric_cols_real))
    
    if len(common_numeric_cols) > 0:
        means_gen = generated_data[common_numeric_cols].mean()
        means_real = real_data[common_numeric_cols].mean()
        stds_gen = generated_data[common_numeric_cols].std()
        stds_real = real_data[common_numeric_cols].std()

        means_diff = np.abs(means_gen - means_real)
        stds_diff = np.abs(stds_gen - stds_real)

        evaluation_results['mean_diff'] = means_diff
        evaluation_results['std_diff'] = stds_diff
    else:
        evaluation_results['mean_diff'] = {}
        evaluation_results['std_diff'] = {}
        
    pearson_corrs = {}
    
    for col in common_numeric_cols:
        try:
            real_col_data = real_data[col].dropna()
            gen_col_data = generated_data[col].dropna()
            
            if len(real_col_data) > 1 and len(gen_col_data) > 1:
                min_len = min(len(real_col_data), len(gen_col_data))
                real_aligned = real_col_data.iloc[:min_len]
                gen_aligned = gen_col_data.iloc[:min_len]
                
                corr, _ = pearsonr(real_aligned, gen_aligned)
                pearson_corrs[col] = corr
        except Exception as e:
            pearson_corrs[col] = np.nan

    evaluation_results['pearson_correlations'] = pearson_corrs

    ks_results = {}
    for col in common_numeric_cols:
        try:
            real_col_data = real_data[col].dropna()
            gen_col_data = generated_data[col].dropna()
            if len(real_col_data) > 1 and len(gen_col_data) > 1:
                ks_stat, ks_p_value = ks_2samp(real_col_data, gen_col_data)
                ks_results[col] = {'ks_stat': ks_stat, 'ks_p_value': ks_p_value}
        except Exception as e:
            ks_results[col] = {'ks_stat': np.nan, 'ks_p_value': np.nan}

    evaluation_results['ks_test'] = ks_results

    feedback = generate_prompt(evaluation_results)
    return feedback

input_df_all = pd.DataFrame()
synthetic_df_all = pd.DataFrame()
synthetic_df_all = generate_data(constraints, "", data, llm, unique_categorical_features, output_parser, TARGET,
                                 N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, N_CLASS, input_df_all,
                                 synthetic_df_all)

# Reverse random words to original values
synthetic_df_all_r = synthetic_df_all.copy()
if params['USE_RANDOM_WORD']:
    for c in mapper_r:
        synthetic_df_all_r[c] = synthetic_df_all_r[c].map(lambda x: mapper_r[c][x] if x in mapper_r[c] else x)

# Save synthetic tabular data
file_name = os.path.join(SYN_DATA_SAVE_DIR, f'{DATA_NAME}_samples_True.csv')
synthetic_df_all_r.to_csv(file_name, index=False)
print('Saved:', file_name)