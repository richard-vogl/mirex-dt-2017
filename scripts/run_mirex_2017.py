import os
import numpy as np
import pickle

from madmom.evaluation.notes import NoteEvaluation, NoteSumEvaluation, NoteMeanEvaluation
from madmom.evaluation.onsets import OnsetEvaluation, OnsetSumEvaluation, OnsetMeanEvaluation

from config import MADMOM_PATH, experiment_base_path, DTCS_PATH, CW_ML_PATH, MATLAB_PATH

"""
folder layout:

mirex-dt-2017
+-eval-set          # MIREX evaluation set - RV has GEN, IDMT, KT, and RBMA - CS has IDMT, (KT?), and MEDLEY
| +-GEN
| | +-annotations
| | +-audio
| +-IDMT            # every dataset folder has an annotations (.txt) and audio (flac/wav/mp3) subfolder
| +-KT
| +-MEDLEY
| +-RBMA
+-public-set        # get these from: https://drive.google.com/drive/folders/0B5QVjGCSDYW1VkZwUEhRSWR5VVk?usp=sharing
| +-2005
| +-GEN
| +-MEDLEY
| +-RBMA
+-results           # subfolder for algorithms contain folder for public (optional) and eval sets
  +-RV1
    +-eval-set      # an evaluation folder contains eval data for all sets and the individual sets
    | +-evaluation
    | +-GEN
    | | +-detections
    | | +-evaluation
    | +-IDMT
    | +-KT
    | +-MEDLEY
    | +-RBMA
    +-public-set
  +-RV2
  +-RV3
+-scripts           # this script
"""

# ==== setup ====
RUN_ON_PUBLIC = False

MADMOM_BIN = os.path.join(MADMOM_PATH, 'bin')
DTCS_BIN = os.path.join(DTCS_PATH, 'bin')

EXEC = 'executable'
PYPATH = 'pypath'
MATPATH = 'matpath'
EXEC_PAT = 'execution_pattern'
BATCH = 'batch'

tolerance = 0.030  # seconds

algo_dict = {
            'RV1': {
                EXEC: os.path.join(MADMOM_BIN, 'DrumTranscriptor'),
                PYPATH: MADMOM_PATH,
                MATPATH: "",
                EXEC_PAT: '%s batch %s -o "%s" -s ".txt"',   # first parameter is the executable (EXEC), second one is the input file (wav), third the output file (detection.txt)
                BATCH: True,
                },
            'RV2': {
                EXEC: os.path.join(MADMOM_BIN, 'DrumTranscriptor'),
                PYPATH: MADMOM_PATH,
                MATPATH: "",
                EXEC_PAT: '%s -m CNN3 batch %s -o "%s" -s ".txt"',
                BATCH: True,
                },
            'RV3': {
                EXEC: os.path.join(MADMOM_BIN, 'DrumTranscriptor'),
                PYPATH: MADMOM_PATH,
                MATPATH: "",
                EXEC_PAT: '%s -m BRNN2 batch %s -o "%s" -s ".txt"',
                BATCH: True,
                },
            'RV4': {
                EXEC: os.path.join(MADMOM_BIN, 'DrumTranscriptor'),
                PYPATH: MADMOM_PATH,
                MATPATH: "",
                EXEC_PAT: '%s -m ENS batch %s -o "%s" -s ".txt"',
                BATCH: True,
            },

            'CS1': {
                EXEC: os.path.join(DTCS_BIN, 'DTCS'),
                PYPATH: DTCS_PATH,
                MATPATH: "",
                EXEC_PAT: '%s -s 0 %s -od "%s"',
                BATCH: True,
            },
            'CS2': {
                EXEC: os.path.join(DTCS_BIN, 'DTCS'),
                PYPATH: DTCS_PATH,
                MATPATH: "",
                EXEC_PAT: '%s -s 1 %s -od "%s"',
                BATCH: True,
            },
            'CS3': {
                EXEC: os.path.join(DTCS_BIN, 'DTCS'),
                PYPATH: DTCS_PATH,
                MATPATH: "",
                EXEC_PAT: '%s -s 2 %s -od "%s"',
                BATCH: True,
            },

            'CW1': {
                EXEC: MATLAB_PATH + " -nojvm -nodisplay -nosplash -nodesktop -r ",
                PYPATH: "",
                MATPATH: os.path.join(CW_ML_PATH),
                EXEC_PAT: "%s \"try, dt_cw({%s}, '%s', 'PfNmf'), catch me, fprintf('%%s / %%s\\n',me.identifier,me.message), end, exit\"",
                # 'PfNmf'
                BATCH: True,
            },
            'CW2': {
                EXEC: MATLAB_PATH + " -nojvm -nodisplay -nosplash -nodesktop -r ",
                PYPATH: "",
                MATPATH: os.path.join(CW_ML_PATH),
                EXEC_PAT: "%s \"try, dt_cw({%s}, '%s', 'Am1'), catch me, fprintf('%%s / %%s\\n',me.identifier,me.message), end, exit\"",
                # 'Am1'
                BATCH: True,
            },
            'CW3': {
                EXEC: MATLAB_PATH + " -nojvm -nodisplay -nosplash -nodesktop -r ",
                PYPATH: "",
                MATPATH: os.path.join(CW_ML_PATH),
                EXEC_PAT: "%s \"try, dt_cw({%s}, '%s', 'Am2'), catch me, fprintf('%%s / %%s\\n',me.identifier,me.message), end, exit\"",
                # 'Am2'
                BATCH: True,
            },
        }

results_base = 'results'

if RUN_ON_PUBLIC:
    dataset_base = 'public-set'
    dataset_names = ['2005', 'GEN', 'MEDLEY', 'RBMA']
else:
    dataset_base = 'eval-set'
    dataset_names = ['IDMT', 'KT', 'GEN', 'RBMA', 'MEDLEY']

audio_path_part = 'audio'
annotations_path_part = 'annotations'
annotations_fixed_path_part = 'annotations_fixed'
detection_path_part = 'detections'
evaluation_path_part = 'evaluation'

NUM_INST = 3


def read_txt_annotations(txt_file, statistics=None):
    with open(txt_file) as f:
        content = f.readlines()

    times = np.ones((len(content), 2)) * -1

    for i_line, line in enumerate(content):
        parts = line.split()
        time = float(parts[0])
        times[i_line][0] = time

        if len(parts) < 2:
            print("warning: No label at line: " + str(i_line) + " in file: " + txt_file)
            if statistics is not None:
                statistics.total_ignored += 1
        else:
            if statistics is not None:
                statistics.total_used += 1

            label = parts[1].strip()
            times[i_line][1] = int(label)

    return times


def filter_inst(entries, num):
    return [entry for entry in entries if entry[1] == num]


results_table = {}

for algo_key in algo_dict:
    algo = algo_dict[algo_key]

    global_eval = []
    global_eval_mean = []
    global_inst_eval = [[] for _ in range(NUM_INST)]
    global_inst_eval_mean = [[] for _ in range(NUM_INST)]

    evaluation_global_path = os.path.join(experiment_base_path, results_base, algo_key, dataset_base, evaluation_path_part)
    if not os.path.exists(evaluation_global_path):
        os.makedirs(evaluation_global_path)

    results_table[algo_key] = {}

    # run through data and make transcription
    for dataset_name in dataset_names:

        cur_ds_path = os.path.join(experiment_base_path, dataset_base, dataset_name)
        cur_op_path = os.path.join(experiment_base_path, results_base, algo_key, dataset_base, dataset_name)

        audio_path = os.path.join(cur_ds_path, audio_path_part)
        detection_path = os.path.join(cur_op_path, detection_path_part)
        annotation_path = os.path.join(cur_ds_path, annotations_path_part)
        evaluation_set_path = os.path.join(cur_op_path, evaluation_path_part)

        if not os.path.exists(detection_path):
            os.makedirs(detection_path)
        if not os.path.exists(evaluation_set_path):
            os.makedirs(evaluation_set_path)

        # ====
        # Create Detections
        # ====
        print('create detections for: ' + algo_key +' for ' + dataset_name)

        if not os.path.exists(audio_path) or len(os.listdir(audio_path)) <= 0:
            print("Skipping: <"+audio_path+"> - no files found")
            continue

        # collect input files
        files = [a_file for a_file in os.listdir(audio_path) if a_file.endswith('.flac') or
                                                                a_file.endswith('.wav') or
                                                                a_file.endswith('.mp3')]

        # setup environment for algo
        if 'PYTHONPATH' in os.environ:
            save_pypath = os.environ['PYTHONPATH']
        else:
            save_pypath = ""
        if 'MATLABPATH' in os.environ:
            save_matpath = os.environ['MATLABPATH']
        else:
            save_matpath = ""
        os.environ['PYTHONPATH'] = algo[PYPATH]
        os.environ['MATLABPATH'] = algo[MATPATH]

        file_list = []
        for cur_file in files:
            out_file = os.path.join(detection_path, os.path.splitext(cur_file)[0] + '.txt')
            if not os.path.exists(out_file): # don't calculate detection again if it exists!
                file_list.append(cur_file)

        # either batch process files or iterate over input files
        if algo[BATCH]:
            file_list_str = ' '.join(['\'' + os.path.join(audio_path, cur_file) + '\'' for cur_file in file_list])

            if len(file_list) > 0:
                print('batch processing: ' + audio_path + ' --> ' + detection_path)
                command = algo[EXEC_PAT] % (algo[EXEC], file_list_str, detection_path)
                os.system(command)
        else:
            for cur_file in file_list:
                in_file = os.path.join(audio_path, cur_file)
                out_file = os.path.join(detection_path, os.path.splitext(cur_file)[0]+'.txt')

                print('processing: ' + in_file + ' --> ' + out_file)
                command = algo[EXEC_PAT] % (algo[EXEC], in_file, out_file)  # run detections
                os.system(command)

        # reset environment
        os.environ['PYTHONPATH'] = save_pypath
        os.environ['MATLABPATH'] = save_matpath

        # ====
        # Run Evaluation
        # ====
        print('run evaluation for: ' + algo_key + ' for ' + dataset_name)

        set_eval = []                                   # evals for single tracks
        set_inst_eval = [[] for _ in range(NUM_INST)]   # evals for single instruments on single tracks

        files = [d_file for d_file in os.listdir(detection_path) if d_file.endswith('.txt')]

        for cur_file in files:
            detection_file = os.path.join(detection_path, os.path.splitext(cur_file)[0] + '.txt')
            annotation_file = os.path.join(annotation_path, os.path.splitext(cur_file)[0] + '.txt')

            detections = read_txt_annotations(detection_file)
            if not os.path.exists(annotation_file):
                continue
            annotations = read_txt_annotations(annotation_file)

            file_eval = NoteEvaluation(detections=detections, annotations=annotations, window=tolerance)
            set_eval.append(file_eval)
            global_eval.append(file_eval)

            file_inst_eval = [[] for _ in range(NUM_INST)]
            for inst in range(NUM_INST):
                file_inst_eval[inst] = OnsetEvaluation(detections=filter_inst(detections, inst),
                                                       annotations=filter_inst(annotations, inst), window=tolerance)
                set_inst_eval[inst].append(file_inst_eval[inst])
                global_inst_eval[inst].append(file_inst_eval[inst])

        set_sum = NoteSumEvaluation(set_eval)    # all instrument, sum evaluation for this set
        set_mean = NoteMeanEvaluation(set_eval)  # all instrument, mean evaluation for this set
        global_eval_mean.append(set_mean)        # append mean eval for this set for global mean

        with open(os.path.join(evaluation_set_path, 'eval.txt'), 'w') as f:
            f.write("Sum Evaluation\n")
            f.write(set_sum.tostring()+"\n")
            f.write("Mean Evaluation\n")
            f.write(set_mean.tostring()+"\n")

        set_inst_sum = [[] for _ in range(NUM_INST)]
        set_inst_mean = [[] for _ in range(NUM_INST)]
        for inst in range(NUM_INST):
            set_inst_sum[inst] = OnsetSumEvaluation(set_inst_eval[inst])    # single instrument, sum evaluation for this set
            set_inst_mean[inst] = OnsetMeanEvaluation(set_inst_eval[inst])  # single instrument, mean evaluation for this set
            global_inst_eval_mean[inst].append(set_inst_mean[inst])         # append single instrument mean for global single instrument mean

            with open(os.path.join(evaluation_set_path, 'eval_inst_'+str(inst)+'.txt'), 'w') as f:
                f.write("Sum Evaluation"+"\n")
                f.write(set_inst_sum[inst].tostring()+"\n")
                f.write("Mean Evaluation"+"\n")
                f.write(set_inst_mean[inst].tostring()+"\n")

        with open(os.path.join(evaluation_set_path, 'eval_data.pkl'), 'w') as f:
            pickle.dump({'set_sum': set_sum, 'set_mean': set_mean, 'set_inst_sum': set_inst_sum,
                         'set_inst_mean': set_inst_mean, 'set_eval': set_eval, 'set_inst_eval': set_inst_eval}, f)

        results_table[algo_key][dataset_name] = {
            'set_mean': set_mean,
            'set_sum': set_sum,
            'inst_mean': set_inst_mean,
            'inst_sum': set_inst_sum,
        }

    global_sum = NoteSumEvaluation(global_eval)         # global all instrument sum evaluation
    global_mean = NoteMeanEvaluation(global_eval_mean)  # global all instrument mean evaluation on means of sets

    with open(os.path.join(evaluation_global_path, 'eval.txt'), 'w') as f:
        f.write("Sum Evaluation"+"\n")
        f.write(global_sum.tostring())
        f.write("Mean Evaluation"+"\n")
        f.write(global_mean.tostring())

    global_inst_sum = [OnsetSumEvaluation(inst) for inst in global_inst_eval]         # global single instrument sum evaluation
    global_inst_mean = [OnsetMeanEvaluation(inst) for inst in global_inst_eval_mean]  # global single instrument mean evaluation on means of sets

    for inst in range(NUM_INST):
        with open(os.path.join(evaluation_global_path, 'eval_inst_' + str(inst) + '.txt'), 'w') as f:
            f.write("Sum Evaluation"+"\n")
            f.write(global_inst_sum[inst].tostring())
            f.write("Mean Evaluation"+"\n")
            f.write(global_inst_mean[inst].tostring())

    with open(os.path.join(evaluation_global_path, 'eval_data.pkl'), 'w') as f:
        pickle.dump({'global_sum': global_sum, 'global_mean': global_mean, 'global_inst_sum': global_inst_sum,
                     'global_inst_mean': global_inst_mean, 'global_eval': global_eval,
                     'global_eval_mean': global_eval_mean, 'global_inst_eval': global_inst_eval,
                     'global_inst_eval_mean': global_inst_eval_mean}, f)

    results_table[algo_key]['global'] = {
        'set_mean': global_mean,
        'set_sum': global_sum,
        'inst_mean': global_inst_mean,
        'inst_sum': global_inst_sum,
    }


# ====
# create table
# ====

tables = {}

for algo_key in results_table.keys():
    algo_results = results_table[algo_key]

    for cur_set in algo_results.keys():
        if cur_set not in tables or len(tables[cur_set]) <= 0:
            tables[cur_set] = "Table for set '"+cur_set+"' : \n"+\
                          "    \t | ALL inst                           | BD                                  | SD                                  | HH                                  |\n"+\
                          "    \t | sum               mean             | sum               mean              | sum               mean              | sum               mean              |\n"+\
                          "algo\t | fm    pr    rc    fm    pr    rc   | fm    pr    rc    fm    pr    rc    | fm    pr    rc    fm    pr    rc    | fm    pr    rc    fm    pr    rc    |\n"+\
                        "---------+------------------------------------+-------------------------------------+-------------------------------------+-------------------------------------+\n"

        fm = algo_results[cur_set]['set_sum'].fmeasure
        pr = algo_results[cur_set]['set_sum'].precision
        rc = algo_results[cur_set]['set_sum'].recall
        fmm = algo_results[cur_set]['set_mean'].fmeasure
        prm = algo_results[cur_set]['set_mean'].precision
        rcm = algo_results[cur_set]['set_mean'].recall

        tables[cur_set] += algo_key+" \t | %1.2f  %1.2f  %1.2f  %1.2f  %1.2f  %1.2f"%(fm, pr, rc, fmm, prm, rcm)

        inst_mean = algo_results[cur_set]['inst_mean']
        inst_sum = algo_results[cur_set]['inst_mean']
        for inst in range(NUM_INST):
            fm = inst_sum[inst].fmeasure
            pr = inst_sum[inst].precision
            rc = inst_sum[inst].recall
            fmm = inst_mean[inst].fmeasure
            prm = inst_mean[inst].precision
            rcm = inst_mean[inst].recall
            tables[cur_set] += " | %1.2f  %1.2f  %1.2f  %1.2f  %1.2f  %1.2f "%(fm, pr, rc, fmm, prm, rcm)

        tables[cur_set]+="| \n"

# print table to console and file
with open(os.path.join(experiment_base_path, results_base, dataset_base+'_results_table.txt'), 'w') as f:
    for table in tables:
        print(tables[table] + "\n")
        f.write(tables[table]+"\n")

