import sys, os, re, subprocess

subject_list_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/advise_task/fitting_actual_data/advise_subject_IDs_prolific.csv'
input_directory = sys.argv[1]
results = sys.argv[2]

if not os.path.exists(results):
    os.makedirs(results)
    print(f"Created results directory {results}")

if not os.path.exists(f"{results}/logs"):
    os.makedirs(f"{results}/logs")
    print(f"Created results-logs directory {results}/logs")

subjects = []
with open(subject_list_path) as infile:
    next(infile)
    for line in infile:
        subjects.append(line.strip())

ssub_path = '/media/labs/rsmith/lab-members/ttakahashi/WellbeingTasks/AdviceTask/RLmodel/toru/run_advise_fit.ssub'

for subject in subjects:
    stdout_name = f"{results}/logs/{subject}-%J.stdout"
    stderr_name = f"{results}/logs/{subject}-%J.stderr"

    jobname = f'advise-fit-{subject}'
    os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {subject} {input_directory} {results}")

    print(f"SUBMITTED JOB [{jobname}]")


    ###python3 /media/labs/rsmith/lab-members/ttakahashi/WellbeingTasks/AdviceTask/RLmodel/toru/runall_advise_fit.py /media/labs/NPC/DataSink/StimTool_Online/WB_Advice /media/labs/rsmith/lab-members/ttakahashi/WellbeingTasks/AdviceTask/resultsforallmodels/model_11
    #squeue -o "%.18i %.9P %.100j %.8u %.2t %.10M %.6D %R"
