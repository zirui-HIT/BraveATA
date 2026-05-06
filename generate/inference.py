import sys
import json
import argparse

sys.path.append('.')


if __name__ == '__main__':
    from utils.execute.executor import Executor
    from utils.generator import generate_with_llm

    args = argparse.ArgumentParser()
    args.add_argument('--data_file', type=str)
    args.add_argument('--data_size', type=int)
    args.add_argument('--dump_file', type=str)
    args.add_argument('--config_file', type=str)
    args.add_argument('--llm_name_or_path', type=str)
    args.add_argument(
        '--task', type=str, choices=['claim_generation', 'theorem_elicitation', 'autoformalization', 'theorem_proving', 'theorem_proving_formal'])
    args.add_argument('--sample_scale', type=int)
    args = args.parse_args()

    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if args.data_size:
            data = data[:args.data_size]
    if 'prediction' in data[0] and args.task in data[0]['prediction']:
        print(f"{args.task} is already in {args.dump_file}")
        exit(0)

    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
        config['n'] = args.sample_scale

    executor = Executor.initialize(args.task)
    inputs = [executor.pack(d) for d in data]
    print("Prompt Packed")
    responses = generate_with_llm(
        args.llm_name_or_path, inputs, config)
    for d, r in zip(data, responses):
        if 'prediction' not in d:
            d['prediction'] = {}
        d['prediction'][args.task] = [{
            "rationale": x[0].strip()
        } for x in r]

    with open(args.dump_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print('Dumped prediction to {}'.format(args.dump_file))
