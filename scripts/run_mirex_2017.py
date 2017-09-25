import os
import numpy as np
import pickle

from madmom.evaluation.notes import NoteEvaluation, NoteSumEvaluation, NoteMeanEvaluation
from madmom.evaluation.onsets import OnsetEvaluation, OnsetSumEvaluation, OnsetMeanEvaluation

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
tolerance = 0.030  # seconds
MADMOM_PATH = '/Users/Rich/python-workspace/madmom_dev'  #TODO: point this to your madmom path for RV algos
experiment_base_path = '/Users/Rich/Desktop/mirex-dt-2017'  #TODO: point this to the DT dataset eval base folder
RUN_ON_PUBLIC = True

MADMOM_BIN = os.path.join(MADMOM_PATH, 'bin')

EXEC = 'executable'
PYPATH = 'path'
EXEC_PAT = 'execution_pattern'
BATCH = 'batch'

algo_dict = {       #TODO: setup algos here!
            'RV1': {
                EXEC: os.path.join(MADMOM_BIN, 'DrumTranscriptor'),
                PYPATH: MADMOM_PATH,
                EXEC_PAT: '%s single "%s" -o "%s"',   # first parameter is the executable (EXEC), second one is the input file (wav), third the output file (detection.txt)
                BATCH: False,
                },
            'RV2': {
                EXEC: os.path.join(MADMOM_BIN, 'DrumTranscriptor'),
                PYPATH: MADMOM_PATH,
                EXEC_PAT: '%s -m CNN3 single "%s" -o "%s"',
                BATCH: False,
                },
            'RV3': {
                EXEC: os.path.join(MADMOM_BIN, 'DrumTranscriptor'),
                PYPATH: MADMOM_PATH,
                EXEC_PAT: '%s -m BRNN2 single "%s" -o "%s"',
                BATCH: False,
                },
            # 'CS1': {
            #     EXEC: 'ADTLib',
            #     PYPATH: MADMOM_PATH,
            #     EXEC_PAT: '%s -i "%s" -o "%s"',
            #     BATCH: False,
            # },
            # 'CS2': {
            #     EXEC: 'ADTLib',
            #     PYPATH: MADMOM_PATH,
            #     EXEC_PAT: '%s -i "%s" -o "%s"',
            #     BATCH: False,
            # },
            # 'CS3': {
            #     EXEC: 'ADTLib',
            #     PYPATH: MADMOM_PATH,
            #     EXEC_PAT: '%s -i "%s" -o "%s"',
            #     BATCH: False,
            # },
            }

results_base = 'results'

if RUN_ON_PUBLIC:
    dataset_base = 'public-set'
    dataset_paths = ['2005', 'GEN', 'MEDLEY', 'RBMA']
else:
    dataset_base = 'eval-set'
    dataset_paths = ['IDMT', 'KT', 'GEN', 'RBMA', 'MEDLEY']

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


for algo_key in algo_dict:
    algo = algo_dict[algo_key]

    global_eval = []
    global_eval_mean = []
    global_inst_eval = [[] for _ in range(NUM_INST)]
    global_inst_eval_mean = [[] for _ in range(NUM_INST)]

    evaluation_global_path = os.path.join(experiment_base_path, results_base, algo_key, dataset_base, evaluation_path_part)
    if not os.path.exists(evaluation_global_path):
        os.makedirs(evaluation_global_path)

    # run through data and make transcription
    for dataset_path in dataset_paths:

        cur_ds_path = os.path.join(experiment_base_path, dataset_base, dataset_path)
        cur_op_path = os.path.join(experiment_base_path, results_base, algo_key, dataset_base, dataset_path)

        audio_path = os.path.join(cur_ds_path, audio_path_part)
        detection_path = os.path.join(cur_op_path, detection_path_part)
        annotation_path = os.path.join(cur_ds_path, annotations_path_part)
        annotation_fixed_path = os.path.join(cur_ds_path, annotations_fixed_path_part)
        evaluation_set_path = os.path.join(cur_op_path, evaluation_path_part)

        if not os.path.exists(detection_path):
            os.makedirs(detection_path)
        if not os.path.exists(annotation_fixed_path):
            os.makedirs(annotation_fixed_path)
        if not os.path.exists(evaluation_set_path):
            os.makedirs(evaluation_set_path)

        # ====
        # Create Detections
        # ====
        print('create detections for: '+algo_key+' for '+dataset_path)

        # collect input files
        files = [a_file for a_file in os.listdir(audio_path) if a_file.endswith('.flac') or
                                                                a_file.endswith('.wav') or a_file.endswith('.mp3')]
        if len(files) <= 0:
            print("Skipping: <"+audio_path+"> - no files found")
            continue

        # setup environment for algo
        save_pypath = os.environ['PYTHONPATH']
        os.environ['PYTHONPATH'] = algo[PYPATH]

        # either batch process files or iterate over input files
        if algo[BATCH]:
            print('batch processing: ' + audio_path + ' --> ' + detection_path)
            file_list = ' '.join([os.path.join(audio_path, cur_file) for cur_file in files])
            command = algo[EXEC_PAT] % (algo[EXEC], file_list, detection_path)
            os.system(command)
        else:
            for cur_file in files:
                in_file = os.path.join(audio_path, cur_file)
                out_file = os.path.join(detection_path, os.path.splitext(cur_file)[0]+'.txt')

                if os.path.exists(out_file):    # don't calculate detection again if it exists!
                    continue

                print('processing: ' + in_file + ' --> ' + out_file)

                command = algo[EXEC_PAT] % (algo[EXEC], in_file, out_file)  # run detections
                os.system(command)

        # reset environment
        os.environ['PYTHONPATH'] = save_pypath

        # ====
        # Run Evaluation
        # ====
        print('run evaluation for: ' + algo_key + ' for ' + dataset_path)

        set_eval = []                                   # evals for single tracks
        set_inst_eval = [[] for _ in range(NUM_INST)]   # evals for single instruments on single tracks

        files = [d_file for d_file in os.listdir(detection_path) if d_file.endswith('.txt')]

        for cur_file in files:
            detection_file = os.path.join(detection_path, os.path.splitext(cur_file)[0] + '.txt')
            annotation_file = os.path.join(annotation_path, os.path.splitext(cur_file)[0] + '.txt')

            detections = read_txt_annotations(detection_file)
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
