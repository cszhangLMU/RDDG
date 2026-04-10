import numpy as np
import pandas as pd
from io import StringIO
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import cross_val_score


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.02)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight, mean=1, std=0.02)
        nn.init.constant_(m.bias, 0)


def compute_eva_score(dataloader, model, eva_epochs, early_window, late_window, optimizer, criterion,
                      device):
    
    model = model.to(device)

    model.train()
    dataset_size = len(dataloader.dataset)
    early_range = early_window[1] - early_window[0]
    late_range = late_window[1] - late_window[0]

    
    l2_errors_per_sample = torch.zeros((dataset_size, early_range + late_range), device=device)

    for epoch in range(eva_epochs):
        for i, (inputs, targets) in enumerate(
                tqdm(dataloader, desc=f'EVA Epoch {epoch + 1}/{eva_epochs}', leave=False)):
            
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if early_window[0] <= epoch < early_window[1] or late_window[0] <= epoch < late_window[1]:
                
                epoch_index = epoch - early_window[0] if epoch < early_window[1] else epoch - late_window[
                    0] + early_range

                
                l2_errors = torch.norm(outputs.detach() - F.one_hot(targets, num_classes=outputs.shape[1]).float(),
                                       dim=1)

                
                for sample_index, l2_error in enumerate(l2_errors):
                    sample_id = i * batch_size + sample_index
                    l2_errors_per_sample[sample_id, epoch_index] = l2_error

    
    variance_early = torch.var(l2_errors_per_sample[:, :early_range], dim=1).cpu().numpy()
    variance_late = torch.var(l2_errors_per_sample[:, early_range:], dim=1).cpu().numpy()
    variance_total = variance_early + variance_late

    return variance_total


def get_prompt_conclass(inital_prompt, numbering, n_samples_per_class, nclass, nset, name_cols):
    prompt = ""
    for i in range(nset):
        prompt += name_cols
        for j in range(nclass):
            prompt += f'{numbering[j]}.\n'
            for k in range(n_samples_per_class):
                prompt += '{' + f'v{i * (n_samples_per_class * nclass) + j * n_samples_per_class + k}' + '}'
            prompt += f'\n'
        prompt += f'\n'
    
    prompt = inital_prompt + prompt
    return prompt


def filtering_categorical(result_df, categorical_features, unique_features):
    org_df = result_df.copy()
    shape_before = org_df.shape

    for column in categorical_features:
        if column == 'Target':
            result_df = result_df[result_df[column].map(lambda x: int(x) in unique_features[column])]
        else:
            result_df = result_df[result_df[column].map(lambda x: x in unique_features[column])]

    if shape_before != result_df.shape:
        for column in categorical_features:
            filtered = org_df[org_df[column].map(lambda x: x not in unique_features[column])]
    return result_df


def parse_prompt2df(one_prompt, split, inital_prompt, col_name):
    one_prompt = one_prompt.replace(inital_prompt, '')
    input_prompt_data = one_prompt.split(split)
    input_prompt_data = [x for x in input_prompt_data if x]
    input_prompt_data = '\n'.join(input_prompt_data)
    input_df = pd.read_csv(StringIO(input_prompt_data), sep=",", header=None, names=col_name)
    input_df = input_df.dropna()
    return input_df


def parse_result(one_prompt, name_cols, col_name, categorical_features, unique_features, filter_flag=True):
    one_prompt = one_prompt.replace(name_cols, '')
    result_df = pd.read_csv(StringIO(one_prompt), sep=",", header=None, names=col_name)
    result_df = result_df.dropna()
    if filter_flag:
        result_df = filtering_categorical(result_df, categorical_features, unique_features)
    return result_df


def get_unique_features(data, categorical_features):
    unique_features = {}
    for column in categorical_features:
        try:
            unique_features[column] = sorted(data[column].unique())
        except:
            unique_features[column] = data[column].unique()
    return unique_features


def get_sampleidx_from_data(unique_features, target, n_samples_total, n_batch, n_samples_per_class, nset, name_cols,
                            data):
    # input sampling
    unique_classes = unique_features[target]
    random_idx_batch_list = []
    target_df_list = []
    for c in unique_classes:
        target_df = data[data[target] == c]
        if len(target_df) < n_samples_total:
            replace_flag = True
        else:
            replace_flag = False
        random_idx_batch = np.random.choice(len(target_df), n_samples_total, replace=replace_flag)
        random_idx_batch = random_idx_batch.reshape(n_batch, nset, 1, n_samples_per_class)
        random_idx_batch_list.append(random_idx_batch)
        target_df_list.append(target_df)
    random_idx_batch_list = np.concatenate(random_idx_batch_list, axis=2)
    return random_idx_batch_list, target_df_list


def get_input_from_idx(target_df_list, random_idx_batch_list, data, n_batch, n_samples_per_class, nset, nclass):
    fv_cols = ('{},' * len(data.columns))[:-1] + '\n'
    # input selection 
    inputs_batch = []
    for batch_idx in range(n_batch):
        inputs = {}
        for i in range(nset):  # 5
            for j in range(nclass):  # 2
                target_df = target_df_list[j]
                for k in range(n_samples_per_class):  # 3
                    idx = random_idx_batch_list[batch_idx, i, j, k]
                    inputs[f'v{i * (n_samples_per_class * nclass) + j * n_samples_per_class + k}'] = fv_cols.format(
                        *target_df.iloc[idx].values
                    )
        inputs_batch.append(inputs)
    return inputs_batch


def make_final_prompt(unique_categorical_features, TARGET, data, template1_prompt,
                      N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, N_CLASS):
    random_idx_batch_list, target_df_list = get_sampleidx_from_data(unique_categorical_features, TARGET,
                                                                    N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS,
                                                                    N_SET, NAME_COLS, data)
    inputs_batch = get_input_from_idx(target_df_list, random_idx_batch_list, data, N_BATCH, N_SAMPLES_PER_CLASS, N_SET,
                                      N_CLASS)
    final_prompt = template1_prompt.batch(inputs_batch)
    # text_contents = [prompt_value.text for prompt_value in final_prompt]
    return final_prompt, inputs_batch


def useThis(one_prompt):
    char = one_prompt[0]
    if char.isdigit() and int(char) in [0, 1, 2, 3, 4]:
        return True, int(char)
    else:
        return False, None


def generate_prompt(evaluation_results):
    
    if any(len(v) == 0 for v in evaluation_results.values()):
        return ""
    
    prompt = "You are generating tabular data. Here is the quality evaluation report:\n\n"

    
    # 1. Mean and standard deviation differences
    prompt += "**Mean and Standard Deviation Differences:**\n"

    if hasattr(evaluation_results['mean_diff'], 'index'):
        for column in evaluation_results['mean_diff'].index:
            prompt += f"   - {column}: Mean diff = {evaluation_results['mean_diff'][column]:.2f}, Std dev diff = {evaluation_results['std_diff'][column]:.2f}.\n"
    elif isinstance(evaluation_results['mean_diff'], dict) and evaluation_results['mean_diff']:
        for column in evaluation_results['mean_diff'].keys():
            prompt += f"   - {column}: Mean diff = {evaluation_results['mean_diff'][column]:.2f}, Std dev diff = {evaluation_results['std_diff'][column]:.2f}.\n"
    else:
        return ""
    prompt += "\n"


    # 2. Column-wise correlations
    prompt += "**Column-wise Correlations (Pearson):**\n"
    if evaluation_results['pearson_correlations']:
        for column, correlation in evaluation_results['pearson_correlations'].items():
            if not pd.isna(correlation):
                prompt += f"   - {column} correlation = {correlation:.2f}.\n"
            else:
                return ""
    else:
        prompt += "   - No correlations available.\n"
    prompt += "\n"

    # 3. Data distribution consistency (K-S test)
    if evaluation_results['ks_test']:
        for column, ks_result in evaluation_results['ks_test'].items():
            if not pd.isna(ks_result['ks_stat']) and not pd.isna(ks_result['ks_p_value']):
                prompt += f"   - {column}: K-S stat = {ks_result['ks_stat']:.2f}, P-value = {ks_result['ks_p_value']:.2f}.\n"
            else:
                return ""
    else:
        return ""
    prompt += "\n"
    

    # Improvement suggestions
    prompt += "\nSuggestions:\n"
    prompt += "- Pay special attention to columns showing large mean and standard deviation differences.\n"
    prompt += "- Ensure better consistency with real data in terms of correlations and distribution.\n"
    prompt += "\nPlease continue generating data and optimize these aspects."

    return prompt

def compute_view_weight(X, y, classifiers, feature_importances, uncertainty, distances):
    scores = {}

    for clf_name, clf in classifiers.items():

        scores[f"{clf_name}_information_gain"] = cross_val_score(clf, X, y, scoring='accuracy', cv=5).mean()

        scores[f"{clf_name}_uncertainty"] = cross_val_score(clf, X, y, scoring='accuracy', cv=5).mean()

        scores[f"{clf_name}_distance"] = cross_val_score(clf, X, y, scoring='accuracy', cv=5).mean()
    return scores


def calculate_combined_scores(X, y, classifiers, feature_importances, uncertainty, distances):

    view_scores = compute_view_weight(X, y, classifiers, feature_importances, uncertainty, distances)

    max_score = max(view_scores.values())
    weights = {view: score / max_score for view, score in view_scores.items()}

    combined_scores = np.zeros(len(feature_importances))
    for i in range(len(feature_importances)):
        combined_scores[i] = (feature_importances[i] * weights.get('XGBoost_information_gain', 0) +
                              uncertainty[i] * weights.get('XGBoost_uncertainty', 0) +
                              distances[i] * weights.get('XGBoost_distance', 0) +
                              feature_importances[i] * weights.get('CatBoost_information_gain', 0) +
                              uncertainty[i] * weights.get('CatBoost_uncertainty', 0) +
                              distances[i] * weights.get('CatBoost_distance', 0) +
                              feature_importances[i] * weights.get('LGBM_information_gain', 0) +
                              uncertainty[i] * weights.get('LGBM_uncertainty', 0) +
                              distances[i] * weights.get('LGBM_distance', 0) +
                              feature_importances[i] * weights.get('GBDT_information_gain', 0) +
                              uncertainty[i] * weights.get('GBDT_uncertainty', 0) +
                              distances[i] * weights.get('GBDT_distance', 0))

    return combined_scores


def select_coreset(X, y, feature_importances, uncertainty, distances, classifiers, num_selected_samples_per_class=100):
    selected_indices = []
    num_classes = len(np.unique(y))

    for class_idx in range(num_classes):

        class_indices = np.where(y == class_idx)[0]
        class_X = X.iloc[class_indices]
        class_y = y.iloc[class_indices]

        combined_scores = calculate_combined_scores(class_X, class_y, classifiers, feature_importances, uncertainty,
                                                    distances)

        top_indices = class_indices[np.argsort(-combined_scores)[:num_selected_samples_per_class]]

        selected_indices.extend(top_indices)

    data_coreset = X.iloc[selected_indices]
    target_coreset = y.iloc[selected_indices]

    return data_coreset, target_coreset
