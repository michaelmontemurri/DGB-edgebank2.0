import re
import pandas as pd

log_paths = ['logs/edgebank_20250421_115632.log']

results = []

for log_path in log_paths:
    with open(log_path, 'r') as f:
        lines = f.readlines()

    current_config = {
        'dataset': None,
        'mem_mode': None,
        'w_mode': None,
        'run': None
    }
    for line in lines:
        line = line.strip()

        # Get mem_mode, w_mode, run 
        if 'INFO:root:network_name:' in line:
            current_config['dataset'] = line.split(':')[-1].strip()

        # Memory and window mode
        elif 'INFO:root:m_mode:' in line:
            current_config['mem_mode'] = line.split(':')[-1].strip()
        elif 'INFO:root:w_mode:' in line:
            current_config['w_mode'] = line.split(':')[-1].strip()
        elif 'INFO:root:Run:' in line:
            current_config['run'] = int(line.split(':')[-1].strip())


        # match metrics
        metric_match = re.match(r'INFO:root:Test statistics: Old nodes -- (\w+): ([\d.]+)', line)
        if metric_match:
            metric, value = metric_match.groups()
            results.append({
                'dataset': current_config.get('dataset'),
                'mem_mode': current_config.get('mem_mode'),
                'w_mode': current_config.get('w_mode', ''),
                'run': current_config.get('run'),
                'metric': metric,
                'value': float(value)
            })



df = pd.DataFrame(results)

# pivot for summary
pivoted = df.pivot_table(
    index=['dataset', 'mem_mode', 'w_mode', 'run'],
    columns='metric',
    values='value'
).reset_index()


summary = pivoted.groupby(['dataset', 'mem_mode', 'w_mode']).agg(['mean', 'std']).round(4)
summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary = summary.reset_index()


pivoted.to_csv('edgebank_raw_results.csv', index=False)
summary.to_csv('edgebank_summary.csv')

print(summary)
