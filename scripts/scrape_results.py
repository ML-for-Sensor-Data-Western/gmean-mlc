import os
import json
import csv
import glob
import re
import argparse

# Define the metrics we want to extract
METRICS = ['MACRO_F1', 'MACRO_F2', 'CIW_F2', 'MAP', 'MACRO_R', 'POSITIVE_F1', 'NEGATIVE_F1']
OPTIMIZATION_METRICS = ['f1', 'f2', 'ap', 'bce', 'loss']

# Function to convert values from 0-1 to 0-100 and round to 2 decimal places
def convert_value(value):
    return round(value * 100, 2)

# Function to extract metrics from a JSON file
def extract_metrics(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        highlights = data.get('Highlights', {})
        metrics = {}
        
        for metric in METRICS:
            # MAP in the data is called mAP or MAP
            if metric == 'MAP' and 'MAP' not in highlights and 'mAP' in highlights:
                metrics[metric] = convert_value(highlights.get('mAP', 0))
            else:
                metrics[metric] = convert_value(highlights.get(metric, 0))
                
        return metrics
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to process each version directory
def process_version(version_dir):
    # Fix for handling version formats like "00_V2"
    base_name = os.path.basename(version_dir)
    version_parts = base_name.split('_')
    if len(version_parts) <= 1:
        version_num = "unknown"
    else:
        # Join all parts after "version_" to handle formats like "00_V2"
        version_num = '_'.join(version_parts[1:])
    
    results = {
        'version': version_num,
        'val': {},
        'test': {}
    }
    
    # Process validation files for each optimization metric
    for opt_metric in OPTIMIZATION_METRICS:
        val_files = glob.glob(os.path.join(version_dir, f"*_{opt_metric}_val_*_0.5.json"))
        if val_files:
            val_file = val_files[0]
            metrics = extract_metrics(val_file)
            if metrics:
                results['val'][opt_metric] = metrics
    
    # Process test files for each optimization metric
    for opt_metric in OPTIMIZATION_METRICS:
        test_files = glob.glob(os.path.join(version_dir, f"*_{opt_metric}_test_*_0.5.json"))
        if test_files:
            test_file = test_files[0]
            metrics = extract_metrics(test_file)
            if metrics:
                results['test'][opt_metric] = metrics
    
    return results

# Main function to scrape all version directories
def main(args):
    results_dir = args.input_dir
    version_dirs = glob.glob(os.path.join(results_dir, 'version_*'))
    
    all_results = []
    for version_dir in version_dirs:
        result = process_version(version_dir)
        all_results.append(result)
    
    # Sort results by version number, trying numeric sort first, falling back to string sort
    def version_sort_key(x):
        version = x['version']
        try:
            # Try to convert to int for numeric sorting
            return (0, int(version))
        except ValueError:
            # If not a simple integer, sort as string but after numeric values
            return (1, version)
    
    all_results.sort(key=version_sort_key)
    
    # Write results to CSV
    with open(args.output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # First row: Version, Split, and Optimum metric headers
        header_row1 = ['Version', 'Split']
        for opt_metric in OPTIMIZATION_METRICS:
            header_row1.append(f'Optimum {opt_metric.upper()}')
            # Add empty cells for the metrics under each optimization
            header_row1.extend([''] * (len(METRICS) - 1))
        writer.writerow(header_row1)
        
        # Second row: Empty for Version and Split, then metric names
        header_row2 = ['', '']
        for _ in OPTIMIZATION_METRICS:
            header_row2.extend(METRICS)
        writer.writerow(header_row2)
        
        # Write data rows for each version
        for result in all_results:
            version = result['version']
            
            # Handle validation results
            if result['val']:
                val_row = [version, 'Val']
                for opt_metric in OPTIMIZATION_METRICS:
                    if opt_metric in result['val']:
                        for metric in METRICS:
                            val_row.append(result['val'][opt_metric].get(metric, ''))
                    else:
                        val_row.extend([''] * len(METRICS))
                writer.writerow(val_row)
            
            # Handle test results if they exist
            if result['test']:
                test_row = [version, 'Test']
                for opt_metric in OPTIMIZATION_METRICS:
                    if opt_metric in result['test']:
                        for metric in METRICS:
                            test_row.append(result['test'][opt_metric].get(metric, ''))
                    else:
                        test_row.extend([''] * len(METRICS))
                writer.writerow(test_row)
    
    print(f"Results have been saved to {args.output_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Scrape metrics from result files')
    parser.add_argument('--input_dir', type=str, default='results', 
                        help='Directory containing the version folders with result files')
    parser.add_argument('--output_path', type=str, default='metrics_results.csv',
                        help='Path to save the output CSV file')
    args = parser.parse_args()
    
    main(args)