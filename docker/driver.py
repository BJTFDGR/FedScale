# Submit job to the remote cluster

import yaml
import sys
import time
import random
import os
import subprocess
import pickle
import datetime


def flatten(d):
    out = {}
    for key, val in d.items():
        if isinstance(val, dict):
            val = [val]
        if isinstance(val, list):
            for subdict in val:
                deeper = flatten(subdict).items()
                out.update({key2: val2 for key2, val2 in deeper})
        else:
            out[key] = val
    return out


def load_yaml_conf(yaml_file):
    with open(yaml_file) as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data


def process_cmd(yaml_file, local=False):

    yaml_conf = load_yaml_conf(yaml_file)

    ps_ip = yaml_conf['ps_ip']
    worker_ips, total_gpus = [], []
    cmd_script_list = []

    executor_configs = "=".join(yaml_conf['worker_ips'])
    for ip_gpu in yaml_conf['worker_ips']:
        ip, gpu_list = ip_gpu.strip().split(':')
        worker_ips.append(ip)
        total_gpus.append(eval(gpu_list))

    time_stamp = datetime.datetime.fromtimestamp(
        time.time()).strftime('%m%d_%H%M%S')
    running_vms = set()
    job_name = 'fedscale_job'
    log_path = './logs'
    submit_user = f"{yaml_conf['auth']['ssh_user']}@" if len(yaml_conf['auth']['ssh_user']) else ""

    job_conf = {'time_stamp': time_stamp,
                'ps_ip': ps_ip,
                }

    for conf in yaml_conf['job_conf']:
        job_conf.update(conf)

    conf_script = ''
    setup_cmd = ''
    if yaml_conf['setup_commands'] is not None:
        setup_cmd += (yaml_conf['setup_commands'][0] + ' && ')
        for item in yaml_conf['setup_commands'][1:]:
            setup_cmd += (item + ' && ')

    cmd_sufix = f" "

    for conf_name in job_conf:
        conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'
        if conf_name == "job_name":
            job_name = job_conf[conf_name]
        if conf_name == "log_path":
            log_path = os.path.join(
                job_conf[conf_name], 'log', job_name, time_stamp)
    # 
    # $FEDSCALE_HOME =: ~/localscratch2/chenboc1/FedScale
    total_gpu_processes = sum([sum(x) for x in total_gpus])
    # =========== Submit job to parameter server ============
    running_vms.add(ps_ip)
    ps_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['aggregator_entry']} {conf_script} --this_rank=0 --num_executors={total_gpu_processes} --executor_configs={executor_configs} "
    # ps_cmd = f" python fedscale/core/aggregation/aggregator.py  --time_stamp=0707_180616 --ps_ip=127.0.1.1 --job_name=femnist --log_path=$FEDSCALE_HOME/benchmark --num_participants=100 --data_set=femnist --data_dir=$FEDSCALE_HOME/benchmark/dataset/data/femnist --data_map_file=$FEDSCALE_HOME/benchmark/dataset/data/femnist/client_data_mapping/train.csv --device_conf_file=$FEDSCALE_HOME/benchmark/dataset/data/device_info/client_device_capacity --device_avail_file=$FEDSCALE_HOME/benchmark/dataset/data/device_info/client_behave_trace --model=shufflenet_v2_x2_0 --gradient_policy=yogi --eval_interval=30 --rounds=1000 --filter_less=21 --num_loaders=2 --yogi_eta=3e-3 --yogi_tau=1e-8 --local_steps=20 --learning_rate=0.05 --batch_size=20 --test_bsz=20 --malicious_factor=4 --use_cuda=True --this_rank=0 --num_executors=20 --executor_configs=127.0.1.1:[0,4,4,4,4,4]"
    with open(f"{job_name}_logging", 'wb') as fout:
        pass
    # shell=True,
    print(f"Starting aggregator on {ps_ip}...")
    with open(f"{job_name}_logging", 'a') as fout:
        if local:
            # subprocess.Popen(f'{ps_cmd}', shell=True,stdout=fout, stderr=fout)
            cmd_sequence=f'{ps_cmd}'
            cmd_sequence=cmd_sequence.split()
            p = subprocess.Popen(cmd_sequence,stdout=fout, stderr=fout)  
            # 'python /home/chenboc1/localscratch2/chenboc1/FedScale/fedscale/core/aggregation/aggregator.py'
        else:
            subprocess.Popen(f'ssh {submit_user}{ps_ip} "{setup_cmd} {ps_cmd}"',
                             shell=True, stdout=fout, stderr=fout)

    time.sleep(10)
    # =========== Submit job to each worker ============
    rank_id = 1
    for worker, gpu in zip(worker_ips, total_gpus):
        running_vms.add(worker)
        print(f"Starting workers on {worker} ...")

        for cuda_id in range(len(gpu)):
            for _ in range(gpu[cuda_id]):
                worker_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['executor_entry']} {conf_script} --this_rank={rank_id} --num_executors={total_gpu_processes} --cuda_device=cuda:{cuda_id} "
                rank_id += 1

                with open(f"{job_name}_logging", 'a') as fout:
                    time.sleep(2)
                    if local:
                        subprocess.Popen(f'{worker_cmd}',
                                         shell=True, stdout=fout, stderr=fout)
                        # cmd_sequence=f'{worker_cmd}'
                        # cmd_sequence=cmd_sequence.split()
                        # p = subprocess.Popen(cmd_sequence,stdout=fout, stderr=fout)                            
                    else:
                        subprocess.Popen(f'ssh {submit_user}{worker} "{setup_cmd} {worker_cmd}"',
                                         shell=True, stdout=fout, stderr=fout)

    # dump the address of running workers
    current_path = os.path.dirname(os.path.abspath(__file__))
    job_name = os.path.join(current_path, job_name)
    with open(job_name, 'wb') as fout:
        job_meta = {'user': submit_user, 'vms': running_vms}
        pickle.dump(job_meta, fout)

    print(f"Submitted job, please check your logs {job_conf['log_path']}/logs/{job_conf['job_name']}/{time_stamp} for status")


def terminate(job_name):

    current_path = os.path.dirname(os.path.abspath(__file__))
    job_meta_path = os.path.join(current_path, job_name)

    if not os.path.isfile(job_meta_path):
        print(f"Fail to terminate {job_name}, as it does not exist")

    with open(job_meta_path, 'rb') as fin:
        job_meta = pickle.load(fin)

    for vm_ip in job_meta['vms']:
        print(f"Shutting down job on {vm_ip}")
        with open(f"{job_name}_logging", 'a') as fout:
            subprocess.Popen(f'ssh {job_meta["user"]}{vm_ip} "python {current_path}/shutdown.py {job_name}"',
                             shell=True, stdout=fout, stderr=fout)

print_help: bool = False
# process_cmd('benchmark/configs/femnist/debug.yml',True)
pass
if len(sys.argv) > 1:
    if sys.argv[1] == 'submit' or sys.argv[1] == 'start':
        process_cmd(sys.argv[2], False if sys.argv[1] == 'submit' else True)
    elif sys.argv[1] == 'stop':
        terminate(sys.argv[2])
    else:
        print_help = True
else:
    print_help = True

if print_help:
    print("\033[0;32mUsage:\033[0;0m\n")
    print("submit $PATH_TO_CONF_YML     # Submit a job")
    print("stop $JOB_NAME               # Terminate a job")
    print()
