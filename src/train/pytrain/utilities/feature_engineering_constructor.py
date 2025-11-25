# 2025 DEPA Foundation
#
# This work is dedicated to the public domain under the CC0 1.0 Universal license.
# To the extent possible under law, DEPA Foundation has waived all copyright and 
# related or neighboring rights to this work. 
# CC0 1.0 Universal (https://creativecommons.org/publicdomain/zero/1.0/)
#
# This software is provided "as is", without warranty of any kind, express or implied,
# including but not limited to the warranties of merchantability, fitness for a 
# particular purpose and noninfringement. In no event shall the authors or copyright
# holders be liable for any claim, damages or other liability, whether in an action
# of contract, tort or otherwise, arising from, out of or in connection with the
# software or the use or other dealings in the software.
#
# For more information about this framework, please visit:
# https://depa.world/training/depa_training_framework/

import pandas as pd
import numpy as np
import json
import os
import ast
import importlib
from typing import Dict, List, Any, Union, Callable
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Unified approved namespace map for feature engineering operations
APPROVED_FE_NAMESPACE_MAP = {
    "pandas": "pandas",
    "numpy": "numpy", 
    "sklearn.preprocessing": "sklearn.preprocessing",
    "sklearn.feature_selection": "sklearn.feature_selection",
    "sklearn.decomposition": "sklearn.decomposition",
    "pd": "pandas",
    "np": "numpy",
}

class FeatureEngineeringConstructor:
    """
    Configurable feature engineering framework with security guardrails.
    Supports declarative feature engineering through JSON configuration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline = self._parse_config(config)
        self.cache = {}
        
    @classmethod
    def load_from_dict(cls, config: Dict[str, Any]):
        """Load feature engineering pipeline from configuration dictionary."""
        return cls(config)
    
    @classmethod
    def load_from_file(cls, config_path: str):
        """Load feature engineering pipeline from JSON configuration file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls(config)
    
    def execute(self, data_sources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the feature engineering pipeline on provided data sources.
        
        Args:
            data_sources: Dictionary mapping data source names to DataFrames or file paths
            
        Returns:
            Dictionary mapping output names to resulting DataFrames
        """
        # Store data_sources for join steps to access
        self._data_sources = data_sources
        results = {}
        
        for step_name, step_config in self.pipeline.items():
            print(f"Executing step: {step_name}")
            
            # Load input data if needed
            inputs = self._resolve_inputs(step_config.get('inputs', {}), data_sources, results)
            
            # Execute the step
            if step_config['type'] == 'load':
                result = self._execute_load_step(step_config, inputs)
            elif step_config['type'] == 'transform':
                result = self._execute_transform_step(step_config, inputs)
            elif step_config['type'] == 'aggregate':
                result = self._execute_aggregate_step(step_config, inputs)
            elif step_config['type'] == 'join':
                result = self._execute_join_step(step_config, inputs)
            elif step_config['type'] == 'preprocess':
                result = self._execute_preprocess_step(step_config, inputs)
            elif step_config['type'] == 'custom':
                result = self._execute_custom_step(step_config, inputs)
            else:
                raise ValueError(f"Unknown step type: {step_config['type']}")
            
            results[step_name] = result
            print(f"  Output shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
        
        return results
    
    def _parse_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the configuration and validate it."""
        if 'pipeline' not in config:
            raise ValueError("Configuration must contain 'pipeline' section")
        
        pipeline = config['pipeline']
        if not isinstance(pipeline, dict):
            raise ValueError("Pipeline must be a dictionary of steps")
        
        # Validate each step
        for step_name, step_config in pipeline.items():
            self._validate_step_config(step_name, step_config)
        
        return pipeline
    
    def _validate_step_config(self, step_name: str, step_config: Dict[str, Any]):
        """Validate a single step configuration."""
        required_fields = ['type']
        for field in required_fields:
            if field not in step_config:
                raise ValueError(f"Step '{step_name}' must have '{field}' field")
        
        step_type = step_config['type']
        valid_types = ['load', 'transform', 'aggregate', 'join', 'preprocess', 'custom']
        if step_type not in valid_types:
            raise ValueError(f"Step '{step_name}' has invalid type '{step_type}'. Must be one of: {valid_types}")
        
        # Type-specific validation
        if step_type == 'load':
            self._validate_load_step(step_name, step_config)
        elif step_type == 'transform':
            self._validate_transform_step(step_name, step_config)
        elif step_type == 'aggregate':
            self._validate_aggregate_step(step_name, step_config)
        elif step_type == 'join':
            self._validate_join_step(step_name, step_config)
        elif step_type == 'preprocess':
            self._validate_preprocess_step(step_name, step_config)
        elif step_type == 'custom':
            self._validate_custom_step(step_name, step_config)
    
    def _validate_load_step(self, step_name: str, step_config: Dict[str, Any]):
        """Validate load step configuration."""
        if 'file_path' not in step_config and 'data_source' not in step_config:
            raise ValueError(f"Load step '{step_name}' must have 'file_path' or 'data_source'")
        
        if 'format' not in step_config:
            raise ValueError(f"Load step '{step_name}' must specify 'format'")
        
        valid_formats = ['csv', 'parquet', 'json', 'hdf5']
        if step_config['format'] not in valid_formats:
            raise ValueError(f"Load step '{step_name}' has invalid format. Must be one of: {valid_formats}")
    
    def _validate_transform_step(self, step_name: str, step_config: Dict[str, Any]):
        """Validate transform step configuration."""
        if 'operation' not in step_config:
            raise ValueError(f"Transform step '{step_name}' must have 'operation'")
        
        valid_operations = [
            'mean', 'sum', 'std', 'var', 'min', 'max', 'count', 'median',
            'normalize', 'standardize', 'robust_scale',
            'pivot', 'unpivot', 'melt',
            'gene_set_enrichment', 'scoring', 'tnse_score',
            'select_columns', 'drop_columns', 'rename_columns',
            'add_constant', 'multiply', 'divide', 'add', 'subtract',
            'log_transform', 'sqrt_transform', 'square_transform',
            'clip', 'replace_values', 'fill_na'
        ]
        
        if step_config['operation'] not in valid_operations:
            raise ValueError(f"Transform step '{step_name}' has invalid operation. Must be one of: {valid_operations}")
    
    def _validate_aggregate_step(self, step_name: str, step_config: Dict[str, Any]):
        """Validate aggregate step configuration."""
        if 'group_by' not in step_config:
            raise ValueError(f"Aggregate step '{step_name}' must have 'group_by'")
        
        if 'aggregations' not in step_config:
            raise ValueError(f"Aggregate step '{step_name}' must have 'aggregations'")
    
    def _validate_join_step(self, step_name: str, step_config: Dict[str, Any]):
        """Validate join step configuration."""
        if 'left' not in step_config or 'right' not in step_config:
            raise ValueError(f"Join step '{step_name}' must have 'left' and 'right' inputs")
        
        if 'on' not in step_config:
            raise ValueError(f"Join step '{step_name}' must specify 'on' column(s)")
    
    def _validate_preprocess_step(self, step_name: str, step_config: Dict[str, Any]):
        """Validate preprocess step configuration."""
        if 'method' not in step_config:
            raise ValueError(f"Preprocess step '{step_name}' must have 'method'")
        
        valid_methods = ['minmax_scale', 'standard_scale', 'robust_scale', 'pca', 'select_k_best', 'select_percentile']
        if step_config['method'] not in valid_methods:
            raise ValueError(f"Preprocess step '{step_name}' has invalid method. Must be one of: {valid_methods}")
    
    def _validate_custom_step(self, step_name: str, step_config: Dict[str, Any]):
        """Validate custom step configuration."""
        if 'function' not in step_config:
            raise ValueError(f"Custom step '{step_name}' must have 'function'")
        
        # Validate function is from approved namespace
        func_path = step_config['function']
        try:
            _resolve_obj_from_approved_path(func_path)
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Custom step '{step_name}' has invalid function path: {e}")
    
    def _resolve_inputs(self, inputs: Dict[str, str], data_sources: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve input references to actual data."""
        resolved = {}
        
        for input_name, input_ref in inputs.items():
            if input_ref in data_sources:
                resolved[input_name] = data_sources[input_ref]
            elif input_ref in results:
                resolved[input_name] = results[input_ref]
            else:
                raise ValueError(f"Input reference '{input_ref}' not found in data sources or previous results")
        
        return resolved
    
    def _execute_load_step(self, step_config: Dict[str, Any], inputs: Dict[str, Any]) -> pd.DataFrame:
        """Execute a data loading step."""
        if 'file_path' in step_config:
            file_path = step_config['file_path']
        else:
            data_source = step_config['data_source']
            file_path = inputs[data_source]
        
        file_format = step_config['format'].lower()
        
        if file_format == 'csv':
            return pd.read_csv(file_path, **step_config.get('params', {}))
        elif file_format == 'parquet':
            return pd.read_parquet(file_path, **step_config.get('params', {}))
        elif file_format == 'json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif file_format == 'hdf5':
            return pd.read_hdf(file_path, **step_config.get('params', {}))
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def _execute_transform_step(self, step_config: Dict[str, Any], inputs: Dict[str, Any]) -> pd.DataFrame:
        """Execute a data transformation step."""
        operation = step_config['operation']
        input_data = list(inputs.values())[0]  # Get first input
        
        if operation == 'mean':
            result = input_data.mean(axis=step_config.get('axis', 0))
            return result.to_frame() if isinstance(result, pd.Series) else result
        elif operation == 'sum':
            result = input_data.sum(axis=step_config.get('axis', 0))
            return result.to_frame() if isinstance(result, pd.Series) else result
        elif operation == 'std':
            result = input_data.std(axis=step_config.get('axis', 0))
            return result.to_frame() if isinstance(result, pd.Series) else result
        elif operation == 'var':
            result = input_data.var(axis=step_config.get('axis', 0))
            return result.to_frame() if isinstance(result, pd.Series) else result
        elif operation == 'min':
            result = input_data.min(axis=step_config.get('axis', 0))
            return result.to_frame() if isinstance(result, pd.Series) else result
        elif operation == 'max':
            result = input_data.max(axis=step_config.get('axis', 0))
            return result.to_frame() if isinstance(result, pd.Series) else result
        elif operation == 'count':
            result = input_data.count(axis=step_config.get('axis', 0))
            return result.to_frame() if isinstance(result, pd.Series) else result
        elif operation == 'median':
            result = input_data.median(axis=step_config.get('axis', 0))
            return result.to_frame() if isinstance(result, pd.Series) else result
        elif operation == 'normalize':
            return self._normalize_data(input_data, step_config)
        elif operation == 'standardize':
            return self._standardize_data(input_data, step_config)
        elif operation == 'robust_scale':
            return self._robust_scale_data(input_data, step_config)
        elif operation == 'pivot':
            return self._pivot_data(input_data, step_config)
        elif operation == 'unpivot':
            return self._unpivot_data(input_data, step_config)
        elif operation == 'melt':
            return self._melt_data(input_data, step_config)
        elif operation == 'gene_set_enrichment':
            return self._gene_set_enrichment(input_data, step_config)
        elif operation == 'scoring':
            return self._scoring(input_data, step_config)
        elif operation == 'tnse_score':
            return self._tnse_score(input_data, step_config)
        elif operation == 'select_columns':
            return self._select_columns(input_data, step_config)
        elif operation == 'drop_columns':
            return self._drop_columns(input_data, step_config)
        elif operation == 'rename_columns':
            return self._rename_columns(input_data, step_config)
        elif operation == 'add_constant':
            return self._add_constant(input_data, step_config)
        elif operation == 'multiply':
            return self._multiply(input_data, step_config)
        elif operation == 'divide':
            return self._divide(input_data, step_config)
        elif operation == 'add':
            return self._add(input_data, step_config)
        elif operation == 'subtract':
            return self._subtract(input_data, step_config)
        elif operation == 'log_transform':
            return self._log_transform(input_data, step_config)
        elif operation == 'sqrt_transform':
            return self._sqrt_transform(input_data, step_config)
        elif operation == 'square_transform':
            return self._square_transform(input_data, step_config)
        elif operation == 'clip':
            return self._clip(input_data, step_config)
        elif operation == 'replace_values':
            return self._replace_values(input_data, step_config)
        elif operation == 'fill_na':
            return self._fill_na(input_data, step_config)
        else:
            raise ValueError(f"Unknown transform operation: {operation}")
    
    def _execute_aggregate_step(self, step_config: Dict[str, Any], inputs: Dict[str, Any]) -> pd.DataFrame:
        """Execute an aggregation step."""
        input_data = list(inputs.values())[0]
        group_by = step_config['group_by']
        aggregations = step_config['aggregations']
        
        if isinstance(group_by, str):
            group_by = [group_by]
        
        return input_data.groupby(group_by).agg(aggregations).reset_index()
    
    def _execute_join_step(self, step_config: Dict[str, Any], inputs: Dict[str, Any]) -> pd.DataFrame:
        """Execute a join step."""
        # For join steps, we need to get the data from the main data_sources
        # This is a special case where the step references data directly
        left_ref = step_config['left']
        right_ref = step_config['right']
        
        # Get data from the main data_sources (passed through the execute method)
        if hasattr(self, '_data_sources'):
            left_data = self._data_sources[left_ref]
            right_data = self._data_sources[right_ref]
        else:
            # Fallback to inputs if data_sources not available
            left_data = inputs.get(left_ref)
            right_data = inputs.get(right_ref)
            
            if left_data is None or right_data is None:
                raise KeyError(f"Data sources '{left_ref}' or '{right_ref}' not found")
        
        on = step_config['on']
        how = step_config.get('how', 'inner')
        
        return left_data.merge(right_data, on=on, how=how)
    
    def _execute_preprocess_step(self, step_config: Dict[str, Any], inputs: Dict[str, Any]) -> pd.DataFrame:
        """Execute a preprocessing step."""
        method = step_config['method']
        input_data = list(inputs.values())[0]
        
        if method == 'minmax_scale':
            scaler = MinMaxScaler(**step_config.get('params', {}))
        elif method == 'standard_scale':
            scaler = StandardScaler(**step_config.get('params', {}))
        elif method == 'robust_scale':
            scaler = RobustScaler(**step_config.get('params', {}))
        elif method == 'pca':
            scaler = PCA(**step_config.get('params', {}))
        elif method == 'select_k_best':
            scaler = SelectKBest(**step_config.get('params', {}))
        elif method == 'select_percentile':
            scaler = SelectPercentile(**step_config.get('params', {}))
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
        
        # Apply preprocessing
        feature_cols = step_config.get('feature_columns', input_data.select_dtypes(include=[np.number]).columns.tolist())
        if feature_cols:
            scaled_data = scaler.fit_transform(input_data[feature_cols])
            result = input_data.copy()
            result[feature_cols] = scaled_data
        else:
            result = input_data.copy()
        
        # Store scaler for later use
        if 'scaler_name' in step_config:
            self.cache[step_config['scaler_name']] = scaler
        
        return result
    
    def _execute_custom_step(self, step_config: Dict[str, Any], inputs: Dict[str, Any]) -> Any:
        """Execute a custom step using approved functions."""
        func_path = step_config['function']
        func = _resolve_obj_from_approved_path(func_path)
        
        params = step_config.get('params', {})
        params.update(inputs)
        
        return func(**params)
    
    # Helper methods for specific transformations
    def _normalize_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Normalize data using MinMax scaling."""
        feature_cols = config.get('feature_columns', data.select_dtypes(include=[np.number]).columns.tolist())
        scaler = MinMaxScaler()
        
        result = data.copy()
        if feature_cols:
            result[feature_cols] = scaler.fit_transform(data[feature_cols])
        
        return result
    
    def _standardize_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Standardize data using z-score normalization."""
        feature_cols = config.get('feature_columns', data.select_dtypes(include=[np.number]).columns.tolist())
        scaler = StandardScaler()
        
        result = data.copy()
        if feature_cols:
            result[feature_cols] = scaler.fit_transform(data[feature_cols])
        
        return result
    
    def _robust_scale_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply robust scaling to data."""
        feature_cols = config.get('feature_columns', data.select_dtypes(include=[np.number]).columns.tolist())
        scaler = RobustScaler()
        
        result = data.copy()
        if feature_cols:
            result[feature_cols] = scaler.fit_transform(data[feature_cols])
        
        return result
    
    def _pivot_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Pivot data from long to wide format."""
        index = config.get('index')
        columns = config.get('columns')
        values = config.get('values')
        
        return data.pivot(index=index, columns=columns, values=values)
    
    def _unpivot_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Unpivot data from wide to long format."""
        id_vars = config.get('id_vars', [])
        value_vars = config.get('value_vars', None)
        var_name = config.get('var_name', 'variable')
        value_name = config.get('value_name', 'value')
        
        return data.melt(id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
    
    def _melt_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Melt data (alias for unpivot)."""
        return self._unpivot_data(data, config)
    
    def _gene_set_enrichment(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Compute gene set enrichment scores."""
        expr_data = data
        target_genes = config.get('target_genes', {})
        
        enrichment = {}
        for drug_name, genes in target_genes.items():
            intersected = [g for g in genes if g in expr_data.columns]
            if len(intersected) == 0:
                gene_avg = pd.Series([0] * expr_data.shape[0], index=expr_data.index)
            else:
                gene_avg = expr_data[intersected].mean(axis=1)
            enrichment[drug_name] = gene_avg
        
        enrichment_df = pd.DataFrame(enrichment)
        enrichment_df.index.name = config.get('index_name', 'cell_id')
        return enrichment_df.reset_index()
    
    def _scoring(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Compute scoring based on markers."""
        expr_data = data
        markers = config.get('markers', {})
        
        scores = pd.DataFrame(index=expr_data.index, columns=markers.keys())
        
        for ctype, marker_genes in markers.items():
            intersected_markers = expr_data.columns.intersection(marker_genes)
            if len(intersected_markers) > 0:
                pos_score = expr_data[intersected_markers].mean(axis=1)
            else:
                pos_score = pd.Series([0] * expr_data.shape[0], index=expr_data.index)
            scores[ctype] = pos_score
        
        scores.index.name = config.get('index_name', 'cell_id')
        suffix = config.get('suffix', '_score')
        scores.columns = [c if c == scores.index.name else f'{c}{suffix}' for c in scores.columns]
        return scores.reset_index()
    
    def _tnse_score(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Compute tNSE scores."""
        enrichment_data = data
        cell_meta = config.get('cell_metadata')
        
        if cell_meta is None:
            raise ValueError("tNSE scoring requires cell_metadata")
        
        # Merge enrichment with cell metadata
        enrichment_with_types = enrichment_data.merge(
            cell_meta.rename(columns={'cell': 'cell_id'}), 
            on='cell_id', 
            how='left'
        )
        
        tnse_scores = {}
        drug_cols = [col for col in enrichment_data.columns if col != 'cell_id']
        
        for cell_type in cell_meta['type'].unique():
            type_cells = enrichment_with_types[enrichment_with_types['type'] == cell_type]
            
            if len(type_cells) == 0:
                tnse_scores[cell_type] = 0
            else:
                n_cells = len(type_cells)
                all_drug_scores = type_cells[drug_cols].values.flatten()
                normalized_scores = (all_drug_scores - all_drug_scores.min()) / (all_drug_scores.max() - all_drug_scores.min() + 1e-8)
                tnse_score = np.sum(normalized_scores) / np.sqrt(n_cells)
                tnse_scores[cell_type] = tnse_score
        
        tnse_df = pd.DataFrame(list(tnse_scores.items()), columns=['cell_type', 'tNSE'])
        return tnse_df
    
    def _select_columns(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Select specific columns."""
        columns = config.get('columns', [])
        return data[columns]
    
    def _drop_columns(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Drop specific columns."""
        columns = config.get('columns', [])
        return data.drop(columns=columns)
    
    def _rename_columns(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Rename columns."""
        mapping = config.get('mapping', {})
        return data.rename(columns=mapping)
    
    def _add_constant(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add a constant value to specified columns."""
        value = config.get('value', 0)
        
        # Handle case where data might be a Series
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        columns = config.get('columns', data.select_dtypes(include=[np.number]).columns.tolist())
        
        result = data.copy()
        result[columns] = result[columns] + value
        return result
    
    def _multiply(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Multiply specified columns by a value."""
        value = config.get('value', 1)
        
        # Handle case where data might be a Series
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        columns = config.get('columns', data.select_dtypes(include=[np.number]).columns.tolist())
        
        result = data.copy()
        result[columns] = result[columns] * value
        return result
    
    def _divide(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Divide specified columns by a value."""
        value = config.get('value', 1)
        
        # Handle case where data might be a Series
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        columns = config.get('columns', data.select_dtypes(include=[np.number]).columns.tolist())
        
        result = data.copy()
        result[columns] = result[columns] / value
        return result
    
    def _add(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add values to specified columns."""
        values = config.get('values', {})
        
        # Handle case where data might be a Series
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        result = data.copy()
        for col, val in values.items():
            if col in result.columns:
                result[col] = result[col] + val
        return result
    
    def _subtract(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Subtract values from specified columns."""
        values = config.get('values', {})
        
        # Handle case where data might be a Series
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        result = data.copy()
        for col, val in values.items():
            if col in result.columns:
                result[col] = result[col] - val
        return result
    
    def _log_transform(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply log transformation."""
        # Handle case where data might be a Series
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        columns = config.get('columns', data.select_dtypes(include=[np.number]).columns.tolist())
        base = config.get('base', np.e)
        
        result = data.copy()
        result[columns] = np.log(result[columns]) / np.log(base)
        return result
    
    def _sqrt_transform(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply square root transformation."""
        # Handle case where data might be a Series
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        columns = config.get('columns', data.select_dtypes(include=[np.number]).columns.tolist())
        
        result = data.copy()
        result[columns] = np.sqrt(result[columns])
        return result
    
    def _square_transform(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply square transformation."""
        # Handle case where data might be a Series
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        columns = config.get('columns', data.select_dtypes(include=[np.number]).columns.tolist())
        
        result = data.copy()
        result[columns] = result[columns] ** 2
        return result
    
    def _clip(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Clip values to specified range."""
        # Handle case where data might be a Series
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        columns = config.get('columns', data.select_dtypes(include=[np.number]).columns.tolist())
        lower = config.get('lower', None)
        upper = config.get('upper', None)
        
        result = data.copy()
        result[columns] = result[columns].clip(lower=lower, upper=upper)
        return result
    
    def _replace_values(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Replace values in specified columns."""
        columns = config.get('columns', data.columns.tolist())
        mapping = config.get('mapping', {})
        
        result = data.copy()
        for col in columns:
            if col in result.columns:
                result[col] = result[col].replace(mapping)
        return result
    
    def _fill_na(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Fill missing values."""
        method = config.get('method', 'forward')
        value = config.get('value', None)
        columns = config.get('columns', data.columns.tolist())
        
        result = data.copy()
        for col in columns:
            if col in result.columns:
                if method == 'forward':
                    result[col] = result[col].fillna(method='ffill')
                elif method == 'backward':
                    result[col] = result[col].fillna(method='bfill')
                elif method == 'value':
                    result[col] = result[col].fillna(value)
                else:
                    raise ValueError(f"Unknown fill method: {method}")
        return result


def _resolve_obj_from_approved_path(path: str):
    """Resolve an attribute object from an approved module path."""
    if not isinstance(path, str):
        raise TypeError("Expected string path")
    
    # Resolve via longest approved prefix
    approved_sorted = sorted(APPROVED_FE_NAMESPACE_MAP.keys(), key=len, reverse=True)
    base = None
    for ns in approved_sorted:
        if path == ns or path.startswith(ns + "."):
            base = ns
            break
    
    if base is None:
        raise ValueError(f"Path '{path}' is not under approved namespaces: {list(APPROVED_FE_NAMESPACE_MAP.keys())}")
    
    provider = APPROVED_FE_NAMESPACE_MAP[base]
    if isinstance(provider, str):
        module = importlib.import_module(provider)
        if path == base:
            raise ValueError(f"Path '{path}' refers to a module, expected a class or function under it")
        remainder = path[len(base) + 1:]
        obj = module
    else:
        if path == base:
            raise ValueError(f"Path '{path}' refers to a namespace root, expected a class or function under it")
        remainder = path[len(base) + 1:]
        obj = provider
    
    for part in remainder.split('.'):
        if part == "":
            raise ValueError(f"Invalid path '{path}'")
        if not hasattr(obj, part):
            raise AttributeError(f"'{obj}' has no attribute '{part}' while resolving '{path}'")
        obj = getattr(obj, part)
    
    return obj
