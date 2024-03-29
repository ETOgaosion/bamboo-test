import os
import re
from dateutil import parser as dateparser
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statistics

PLOT_EACH = True
PLOT_AVG = True

# valid file
valid_file = re.compile(r'node_\d+\.log')
# judge the log line
is_log_parser = re.compile(r'\[ \d{2,}\|\d{2,} \]')
# extract the time
time_parser = re.compile(r'(?P<time>\d{4,}-\d{2,}-\d{2,} \d{2,}:\d{2,}:\d{2,}.\d+)')
# extract the batch operation
start_batch_parser = re.compile(r'START BATCH (?P<batchid>\d+)')
finish_batch_parser = re.compile(r'FINISH BATCH (?P<batchid>\d+) took (?P<batchtime>\S+) s')
start_local_model_train_parser = re.compile(r'START LOCAL MODEL TRAIN (?P<globalstep>\d+)')
finish_local_model_train_parser = re.compile(r'FINISH LOCAL MODEL TRAIN (?P<globalstep>\d+)')
failure_node_detect_parser = re.compile(r'\[Engine\] Signal handler called with signal 15')
# extract the failure
failure_detect_parser = re.compile(r'FAILURES')
# extract the exception
start_next_stage_exception_parser = re.compile(r'START NextStageException fallback schedule (?P<globalstep>\d+)')
finish_next_stage_exception_parser = re.compile(r'FINISH NextStageException fallback schedule (?P<globalstep>\d+)')
start_prev_stage_exception_parser = re.compile(r'START PrevStageException fallback schedule (?P<globalstep>\d+)')
finish_prev_stage_exception_parser = re.compile(r'FINISH PrevStageException fallback schedule (?P<globalstep>\d+)')
# extract the reconfigure
start_reconfigure_parser = re.compile(r'START RECONFIGURE (?P<globalstep>\d+)')
finish_save_shadow_note_parser = re.compile(r'FINISH SAVE SHADOW NODE STATE (?P<globalstep>\d+)')
start_reconfigure_cluster_parser = re.compile(r'START RECONFIGURE CLUSTER and TRANSFER LAYERS (?P<globalstep>\d+)')
finish_reconfigure_parser = re.compile(r'FINISH RECONFIGURE (?P<globalstep>\d+)')

raw_data_tags = ['start_batch_times', 'finish_batch_times', 'start_local_model_train_times', 'finish_local_model_train_times', 'batch_times', 'start_next_stage_exception_times', 'finish_next_stage_exception_times', 'start_prev_stage_exception_times', 'finish_prev_stage_exception_times', 'start_reconfigure_times', 'finish_save_shadow_note_times', 'start_reconfigure_cluster_times', 'finish_reconfigure_times', 'fail_point']
mid_data_tags = ['delta_batch_times', 'delta_local_model_train_times', 'delta_next_stage_exception_times', 'delta_prev_stage_exception_times', 'delta_reconfigure_times', 'delta_reconfigure_cluster_times', 'delta_save_shadow_note_times', 'fail_point', 'maxi', 'mini']
tags = ['delta_local_model_train_time', 'delta_next_stage_exception_time', 'delta_prev_stage_exception_time', 'delta_save_shadow_note_time', 'delta_reconfigure_cluster_time', 'delta_reconfigure_time', 'delta_batch_time']

def res_parser(file):
    print(f'processing: {file}')
    print
    raw_data = {
        'start_batch_times': {},
        'finish_batch_times': {},
        'start_local_model_train_times': {},
        'finish_local_model_train_times': {},
        'batch_times': {},
        'start_next_stage_exception_times': {},
        'finish_next_stage_exception_times': {},
        'start_prev_stage_exception_times': {},
        'finish_prev_stage_exception_times': {},
        'start_reconfigure_times': {},
        'finish_save_shadow_note_times': {},
        'start_reconfigure_cluster_times': {},
        'finish_reconfigure_times': {}
    }
    append_points = []
    fail_point = -1
    with open(file, 'r') as fp:
        for line in fp.readlines():
            if fail_point == -1 and (failure_node_detect_parser.match(line) or failure_detect_parser.search(line)):
                fail_point = len(raw_data['start_batch_times'])
            if is_log_parser.match(line) is None:
                continue
            time = time_parser.search(line)
            start_batch = start_batch_parser.search(line)
            if start_batch:
                batchid = int(start_batch.group('batchid'))
                if not raw_data['start_batch_times']:
                    if fail_point != -1:
                        fail_point += batchid
                raw_data['start_batch_times'][batchid] = dateparser.parse(time.group('time'))
                continue
            finish_batch = finish_batch_parser.search(line)
            if finish_batch:
                batchid = int(finish_batch.group('batchid'))
                raw_data['finish_batch_times'][batchid] = dateparser.parse(time.group('time'))
                raw_data['batch_times'][batchid] = float(finish_batch.group('batchtime'))
                continue
            start_local_model_train = start_local_model_train_parser.search(line)
            if start_local_model_train:
                globalstep = int(start_local_model_train.group('globalstep'))
                raw_data['start_local_model_train_times'][globalstep] = dateparser.parse(time.group('time'))
                continue
            finish_local_model_train = finish_local_model_train_parser.search(line)
            if finish_local_model_train:
                globalstep = int(finish_local_model_train.group('globalstep'))
                raw_data['finish_local_model_train_times'][globalstep] = dateparser.parse(time.group('time'))
                continue
            start_next_stage_exception = start_next_stage_exception_parser.search(line)
            if start_next_stage_exception:
                globalstep = int(start_next_stage_exception.group('globalstep'))
                raw_data['start_next_stage_exception_times'][globalstep] = dateparser.parse(time.group('time'))
                continue
            finish_next_stage_exception = finish_next_stage_exception_parser.search(line)
            if finish_next_stage_exception:
                globalstep = int(finish_next_stage_exception.group('globalstep'))
                raw_data['finish_next_stage_exception_times'][globalstep] = dateparser.parse(time.group('time'))
                continue
            start_prev_stage_exception = start_prev_stage_exception_parser.search(line)
            if start_prev_stage_exception:
                globalstep = int(start_prev_stage_exception.group('globalstep'))
                raw_data['start_prev_stage_exception_times'][globalstep] = dateparser.parse(time.group('time'))
                continue
            finish_prev_stage_exception = finish_prev_stage_exception_parser.search(line)
            if finish_prev_stage_exception:
                globalstep = int(finish_prev_stage_exception.group('globalstep'))
                raw_data['finish_prev_stage_exception_times'][globalstep] = dateparser.parse(time.group('time'))
                continue
            start_reconfigure = start_reconfigure_parser.search(line)
            if start_reconfigure:
                globalstep = int(start_reconfigure.group('globalstep'))
                raw_data['start_reconfigure_times'][globalstep] = dateparser.parse(time.group('time'))
                append_points.append(globalstep)
                continue
            finish_save_shadow_note = finish_save_shadow_note_parser.search(line)
            if finish_save_shadow_note:
                globalstep = int(finish_save_shadow_note.group('globalstep'))
                raw_data['finish_save_shadow_note_times'][globalstep] = dateparser.parse(time.group('time'))
                continue
            start_reconfigure_cluster = start_reconfigure_cluster_parser.search(line)
            if start_reconfigure_cluster:
                globalstep = int(start_reconfigure_cluster.group('globalstep'))
                raw_data['start_reconfigure_cluster_times'][globalstep] = dateparser.parse(time.group('time'))
                continue
            finish_reconfigure = finish_reconfigure_parser.search(line)
            if finish_reconfigure:
                globalstep = int(finish_reconfigure.group('globalstep'))
                raw_data['finish_reconfigure_times'][globalstep] = dateparser.parse(time.group('time'))
                continue
    assert len(raw_data['start_batch_times']) == len(raw_data['finish_batch_times']) == len(raw_data['batch_times']) == len(raw_data['start_local_model_train_times']) == len(raw_data['finish_local_model_train_times']) and len(raw_data['start_next_stage_exception_times']) == len(raw_data['finish_next_stage_exception_times']) and len(raw_data['start_prev_stage_exception_times']) == len(raw_data['finish_prev_stage_exception_times']) and len(raw_data['start_reconfigure_times']) == len(raw_data['finish_save_shadow_note_times']) == len(raw_data['start_reconfigure_cluster_times']) == len(raw_data['finish_reconfigure_times']), f'{len(raw_data["start_batch_times"])} {len(raw_data["finish_batch_times"])} {len(raw_data["batch_times"])} {len(raw_data["start_local_model_train_times"])} {len(raw_data["finish_local_model_train_times"])} {len(raw_data["start_next_stage_exception_times"])} {len(raw_data["finish_next_stage_exception_times"])} {len(raw_data["start_prev_stage_exception_times"])} {len(raw_data["finish_prev_stage_exception_times"])} {len(raw_data["start_reconfigure_times"])} {len(raw_data["finish_save_shadow_note_times"])} {len(raw_data["start_reconfigure_cluster_times"])} {len(raw_data["finish_reconfigure_times"])}'
    assert not ((len(append_points) > 0) & (fail_point != -1)), f'{len(append_points)} {fail_point}'
    return raw_data, append_points, fail_point

def time2int(td):
    return td.total_seconds() * 1000

def pre_handle_data(raw_data):
    mid_data = {
        'delta_batch_times': {},
        'delta_local_model_train_times': {},
        'delta_next_stage_exception_times': {},
        'delta_prev_stage_exception_times': {},
        'delta_reconfigure_times': {},
        'delta_reconfigure_cluster_times': {},
        'delta_save_shadow_note_times': {}
    }
    for k, v in raw_data['start_batch_times'].items():
        mid_data['delta_batch_times'][k] = time2int(raw_data['finish_batch_times'][k] - v)
        mid_data['delta_local_model_train_times'][k] = time2int(raw_data['finish_local_model_train_times'][k] - raw_data['start_local_model_train_times'][k])
        raw_data['batch_times'][k] = raw_data['batch_times'][k] * 1000
    for k, v in raw_data['start_next_stage_exception_times'].items():
        mid_data['delta_next_stage_exception_times'][k] = time2int(raw_data['finish_next_stage_exception_times'][k] - v)
    for k, v in raw_data['start_prev_stage_exception_times'].items():
        mid_data['delta_prev_stage_exception_times'][k] = time2int(raw_data['finish_prev_stage_exception_times'][k] - v)
    for k, v in raw_data['start_reconfigure_times'].items():
        mid_data['delta_reconfigure_times'][k] = time2int(raw_data['finish_reconfigure_times'][k] - v)
        mid_data['delta_reconfigure_cluster_times'][k] = time2int(raw_data['finish_reconfigure_times'][k] - raw_data['start_reconfigure_cluster_times'][k])
        mid_data['delta_save_shadow_note_times'][k] = time2int(raw_data['finish_save_shadow_note_times'][k] - v)
    data = {}
    for k, v in mid_data['delta_batch_times'].items():
        data[k] = {'delta_batch_time': v, 'delta_local_model_train_time': mid_data['delta_local_model_train_times'][k], 'batch_time': raw_data['batch_times'][k]}
        if k in mid_data['delta_next_stage_exception_times']:
            data[k]['delta_next_stage_exception_time'] = mid_data['delta_next_stage_exception_times'][k]
        else:
            data[k]['delta_next_stage_exception_time'] = 0
        if k in mid_data['delta_prev_stage_exception_times']:
            data[k]['delta_prev_stage_exception_time'] = mid_data['delta_prev_stage_exception_times'][k] + data[k]['delta_next_stage_exception_time']
        else:
            data[k]['delta_prev_stage_exception_time'] = data[k]['delta_next_stage_exception_time']
        if k in mid_data['delta_reconfigure_times']:
            data[k]['delta_reconfigure_time'] = mid_data['delta_reconfigure_times'][k] + data[k]['delta_prev_stage_exception_time']
            data[k]['delta_reconfigure_cluster_time'] = mid_data['delta_reconfigure_cluster_times'][k] + data[k]['delta_reconfigure_time']
            data[k]['delta_save_shadow_note_time'] = mid_data['delta_save_shadow_note_times'][k] + data[k]['delta_reconfigure_cluster_time']
    return mid_data, data, max(mid_data['delta_batch_times'].values()), min(mid_data['delta_batch_times'].values())

def handle_data(pre_handled_data, append_points, fail_point):
    data = []
    dalta_batch_times, delta_local_model_train_times = [], []
    if append_points:
        last_point = sorted(pre_handled_data['delta_batch_times'].keys())[0]
        for point in append_points:
            for i in range(last_point, point):
                dalta_batch_times.append(pre_handled_data['delta_batch_times'][i])
                delta_local_model_train_times.append(pre_handled_data['delta_local_model_train_times'][i])
            data.append({
                'delta_batch_time': statistics.mean(dalta_batch_times),
                'delta_local_model_train_time': statistics.mean(delta_local_model_train_times)
            })
            data.append({
                'delta_batch_time': pre_handled_data['delta_batch_times'][point],
                'delta_local_model_train_time': pre_handled_data['delta_local_model_train_times'][point],
                'delta_reconfigure_time': pre_handled_data['delta_reconfigure_times'][point],
                'delta_reconfigure_cluster_time': pre_handled_data['delta_reconfigure_cluster_times'][point],
                'delta_save_shadow_note_time': pre_handled_data['delta_save_shadow_note_times'][point]
            })
            last_point = point + 1
            dalta_batch_times, delta_local_model_train_times = [], []
    elif fail_point != -1:
        # first normal data
        for i in range(sorted(pre_handled_data['delta_batch_times'].keys())[0], fail_point):
            dalta_batch_times.append(pre_handled_data['delta_batch_times'][i])
            delta_local_model_train_times.append(pre_handled_data['delta_local_model_train_times'][i])
        data.append({
            'delta_batch_time': statistics.mean(dalta_batch_times),
            'delta_local_model_train_time': statistics.mean(dalta_batch_times)
        })
        fail_point = 1
        delta_next_stage_exception_times, delta_prev_stage_exception_times = [], []
        for i in range(fail_point, len(pre_handled_data['delta_batch_times'])):
            dalta_batch_times.append(pre_handled_data['delta_batch_times'][i])
            delta_local_model_train_times.append(pre_handled_data['delta_local_model_train_times'][i])
            if i in pre_handled_data['delta_next_stage_exception_times']:
                delta_next_stage_exception_times.append(pre_handled_data['delta_next_stage_exception_times'][i])
            if i in pre_handled_data['delta_prev_stage_exception_times']:
                delta_prev_stage_exception_times.append(pre_handled_data['delta_prev_stage_exception_times'][i])
        data.append({
            'delta_batch_time': statistics.mean(dalta_batch_times),
            'delta_local_model_train_time': statistics.mean(delta_local_model_train_times)
        })
        if delta_next_stage_exception_times:
            data[-1]['delta_next_stage_exception_time'] = statistics.mean(delta_next_stage_exception_times)
        if delta_prev_stage_exception_times:
            data[-1]['delta_prev_stage_exception_time'] = statistics.mean(delta_prev_stage_exception_times)
    else:
        for i in sorted(pre_handled_data['delta_batch_times'].keys()):
            dalta_batch_times.append(pre_handled_data['delta_batch_times'][i])
            delta_local_model_train_times.append(pre_handled_data['delta_local_model_train_times'][i])
        data.append({
            'delta_batch_time': statistics.mean(dalta_batch_times),
            'delta_local_model_train_time': statistics.mean(delta_local_model_train_times)
        })
    return data, fail_point

def plot_each(target, data, maxi, mini, fail_point):
    c_success, c_fail = ['blue', 'violet', 'salmon', 'orange', 'cyan', 'goldenrod', 'magenta'], []
    zorder, alpha, handles = [], [], []
    for i in range(7):
        c_fail.append('dark' + c_success[i])
        zorder.append(8 - i)
        if i != 6:
            alpha.append(0.6)
        else:
            alpha.append(1)
        handles.append(mpatches.Patch(color=c_success[i], label=tags[i]))
        handles.append(mpatches.Patch(color=c_fail[i], label=tags[i] + " fail"))
    for k, v in data.items():
        if k >= fail_point:
            for j in range(7):
                if tags[j] in v:
                    # print(k, j, v[tags[j]], c_fail[j], zorder[j], alpha[j])
                    plt.bar(k, v[tags[j]], color=c_fail[j], zorder=zorder[j], alpha=alpha[j])
        else:
            for j in range(7):
                if tags[j] in v:
                    # print(k, j, v[tags[j]], c_success[j], zorder[j], alpha[j])
                    plt.bar(k, v[tags[j]], color=c_success[j], zorder=zorder[j], alpha=alpha[j])
    plt.xlabel('Batch Number')
    plt.ylim(mini - 200, maxi + 200)
    plt.ylabel('Execution Time (ms)')
    plt.legend(handles=handles, fontsize=5)
    plt.savefig(target)
    plt.close()

def plot_avgs(axes, title, data, maxi, mini, fail_point):
    c_success, c_fail = ['blue', 'violet', 'salmon', 'orange', 'cyan', 'goldenrod', 'magenta'], []
    zorder, alpha, handles = [], [], []
    for i in range(7):
        c_fail.append('dark' + c_success[i])
        zorder.append(8 - i)
        if i != 6:
            alpha.append(0.6)
        else:
            alpha.append(1)
        handles.append(mpatches.Patch(color=c_success[i], label=tags[i]))
        handles.append(mpatches.Patch(color=c_fail[i], label=tags[i] + ' fail'))
    ticks = []
    for k, v in enumerate(data):
        if fail_point != -1 and k >= fail_point:
            for j in range(7):
                if tags[j] in v:
                    # print(k, j, v[tags[j]], tags[j], c_fail[j], zorder[j], alpha[j])
                    bar = axes.bar(k, v[tags[j]], color=c_fail[j], zorder=zorder[j], alpha=alpha[j])
                    axes.bar_label(bar, label_type='edge', padding=3, zorder=99)
        else:
            for j in range(7):
                if tags[j] in v:
                    # print(k, j, v[tags[j]], tags[j], c_success[j], zorder[j], alpha[j])
                    bar = axes.bar(k, v[tags[j]], color=c_success[j], zorder=zorder[j], alpha=alpha[j])
                    axes.bar_label(bar, label_type='edge', padding=3, zorder=99)
        ticks.append('Stage ' + str(k))
    axes.set_xticks(range(len(data)), ticks)
    axes.set_xlabel('Stage')
    axes.set_ylim(mini - 200, maxi + 200)
    axes.set_ylabel('Execution Time (ms)')
    axes.set_title(title)
    return handles

dirs = os.listdir('res')
while '.DS_Store' in dirs: dirs.remove('.DS_Store')
while '.gitignore' in dirs: dirs.remove('.gitignore')

if PLOT_EACH:
    for diri in dirs:
        files = os.listdir('res/' + diri)
        while '.DS_Store' in files: files.remove('.DS_Store')
        for item in files:
            raw_data, append_points, fail_point = res_parser('res/' + diri + '/' + item)
            
            _, data, maxi, mini = pre_handle_data(raw_data)
            
            plot_each('fig/' + diri + '/' + item[:-4] + '.png', data, maxi, mini, fail_point)

if PLOT_AVG:
    for diri in dirs:
        files = os.listdir('res/' + diri)
        while '.DS_Store' in files: files.remove('.DS_Store')
        while '.gitignore' in files: files.remove('.gitignore')
        files = sorted(files)
        fig, axes = plt.subplots(1, len(files))
        fig.set_size_inches(5 * len(files), 10)
        fig.suptitle(f'{diri} Execution Time')
        handles = []
        for idx, item in enumerate(files):
            raw_data, append_points, fail_point = res_parser('res/' + diri + '/' + item)
            
            mid_data, _, maxi, mini = pre_handle_data(raw_data)
            
            data, fail_point = handle_data(mid_data, append_points, fail_point)
            
            handles = plot_avgs(axes[idx], item, data, maxi, mini, fail_point)
        plt.legend(handles=handles, fontsize=10)
        plt.tight_layout(pad=1, w_pad=1, h_pad=2)
        plt.savefig('fig/' + diri + '/nodes.png')
        plt.close()