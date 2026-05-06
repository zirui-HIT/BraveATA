import sys
import json
import argparse

sys.path.append('.')


if __name__ == '__main__':
    from utils.execute.executor import Executor

    args = argparse.ArgumentParser()
    args.add_argument('--data_file', type=str)
    args.add_argument('--dump_file', type=str)
    args.add_argument('--data_size', type=int)
    args.add_argument(
        '--task', type=str, choices=['claim_generation', 'theorem_elicitation', 'autoformalization', 'theorem_proving', 'theorem_proving_formal'])
    args = args.parse_args()

    with open(args.data_file, 'r') as f:
        data = json.load(f)
        if args.data_size is not None:
            data = data[:args.data_size]
        if all('evaluation' in d['prediction'][args.task][0] for d in data):
            print(f'{args.task} for {args.data_file} evaluated, skip evaluation')
            exit(0)

    executor = Executor.initialize(args.task, activate_eval=True)
    for d in data:
        for p in d['prediction'][args.task]:
            p['answer'] = executor.extract(p['rationale'])
    data = executor.evaluate_multi(data, dump_file=args.dump_file)

    with open(args.dump_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
