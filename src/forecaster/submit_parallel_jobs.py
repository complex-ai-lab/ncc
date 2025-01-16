import os
import argparse

job_ids = [22051 + i for i in range(5)]

OURMETHOD = 'NCC'
baseline_names = ['nexcp', 'faci', 'cfrnn', 'aci', 'pid']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval', action='store_true')
    parser.add_argument('-c', '--copy', action='store_true')
    parser.add_argument('-b', '--baseline', action='store_true')
    args = parser.parse_args()
    
    if args.eval:
        for job_id in job_ids:
            print(f'Start evaluating job {job_id}')
            os.system(f'python eval_cp.py -m=e2ecp -i={job_id} -u -p=flu -c -o')
            if args.baseline:
                for method in baseline_names:
                    os.system(f'python eval_cp.py -m={method} -i={job_id} -s=1 -c -o')
            print(f'###### Finish evaluating job {job_id}')
    elif args.copy:
        # remove files in results2share
        os.system('rm ../../results/results2share/*')
        # copy files to results2share from results
        for jobid in job_ids:
            os.system(f'cp ../../results/{jobid}*.pkl ../../results/results2share/')
    else:
        for job_id in job_ids:
            os.system(f'python parallel_job_runs.py -i={job_id}')
        
