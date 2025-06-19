import os
import ujson as json
from tqdm import tqdm
from llama_index.llms.ollama import Ollama


def extract_triplets(llm, ctx):
    """从文本中提取三元组"""
    query = (
        f'Extract triplets informative from the text following the examples. '
        f'Make sure the triplet texts are only directly from the given text! '
        f'Complete directly and strictly following the instructions without any additional words, line break nor space!\n'
        f'{"-" * 20}\n'
        f'Text: Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.\n'
        f'Triplets:<Scott Derrickson##born in##1966>$$<Scott Derrickson##nationality##America>$$<Scott Derrickson##occupation##director>$$<Scott Derrickson##occupation##screenwriter>$$<Scott Derrickson##occupation##producer>$$\n'
        f'{"-" * 20}\n'
        f'Text: A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. Shirley Temple was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.\n'
        f'Triplets:<A Kiss for Corliss##cast member##Shirley Temple>$$<Shirley Temple##served as##Chief of Protocol>$$\n'
        f'{"-" * 20}\n'
        f'Text: {ctx}\n'
        f'Triplets:'
    )

    resp = llm.complete(query)
    resp_text = resp.text.strip()

    triplets = set()
    triplet_texts = resp_text.split('$$')

    for triplet_text in triplet_texts:
        triplet_text = triplet_text.strip()
        if not triplet_text or len(triplet_text) < 5:
            continue

        # 清理三元组格式
        if triplet_text.startswith('<') and triplet_text.endswith('>'):
            triplet_text = triplet_text[1:-1]

        tokens = triplet_text.split('##')
        if len(tokens) != 3:
            continue

        h, r, t = (token.strip() for token in tokens)

        # 过滤无效三元组
        if any(invalid in h or invalid in t
               for invalid in ['no ', 'unknown', 'No ', 'Unknown', 'null', 'Null', 'NULL', 'NO']):
            continue

        if h == t or (r not in ctx and t not in ctx):
            continue

        triplets.add((h, r, t))

    return [[h, r, t] for h, r, t in triplets]


def main():
    # 配置路径
    data_path = '../../data/hotpotqa/hotpot_dev_fullwiki_v1.json'
    out_dir = '../../data/hotpotqa/kgs/extract_subkgs1'
    os.makedirs(out_dir, exist_ok=True)

    # 加载数据
    with open(data_path, 'r') as f:
        data = json.load(f)

    # 初始化LLM
    llm = Ollama(model='llama3:8b-instruct-fp16', request_timeout=120)

    # 创建实体处理状态跟踪器
    processed_entities = set()
    entity_files = {}
    count = 0

    # 第一遍：收集所有实体
    print("Scanning entities...")
    for sample in tqdm(data):
        for ctx in sample['context']:
            ent = ctx[0]
            if ent not in processed_entities:
                processed_entities.add(ent)
                # 检查实体文件是否存在
                ent_file = f'{ent.replace("/", "_")}.json'
                file_path = os.path.join(out_dir, ent_file)

                if os.path.exists(file_path):
                    # 文件已存在，加载已有数据
                    with open(file_path, 'r') as f:
                        entity_files[ent] = json.load(f)
                else:
                    # 文件不存在，初始化空字典
                    entity_files[ent] = {}

    # 第二遍：处理实体
    print("Processing entities...")
    for sample in tqdm(data):
        for ctx in sample['context']:
            ent = ctx[0]
            ent_data = entity_files.get(ent, {})
            updated = False

            for i in range(len(ctx[1])):
                # 跳过已处理的段落
                if str(i) in ent_data:
                    continue

                # 准备上下文文本
                ctx_text = ctx[1][i] if i == 0 else f'{ent}: {ctx[1][i]}'

                # 提取三元组
                ext_triplets = extract_triplets(llm, ctx_text)

                if ext_triplets:
                    ent_data[str(i)] = ext_triplets
                    updated = True

            # 如果有更新，保存到内存
            if updated:
                entity_files[ent] = ent_data

    # 保存更新到文件
    print("Saving results...")
    for ent, ent_data in tqdm(entity_files.items()):
        ent_file = f'{ent.replace("/", "_")}.json'
        file_path = os.path.join(out_dir, ent_file)

        with open(file_path, 'w') as f:
            json.dump(ent_data, f)
            count += 1

    print(f'Processed entities: {count}')


if __name__ == '__main__':
    main()