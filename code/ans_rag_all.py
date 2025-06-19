import os
import copy
import ujson as json
import argparse
import re
import random
import time
import string
import networkx as nx
import logging
from tqdm import tqdm
from typing import List, Dict, Optional, Set, Sequence, Any, Callable, Generator, Type, cast, AsyncGenerator
from FlagEmbedding import FlagReranker
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.bridge.pydantic import Field, BaseModel
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.settings import Settings as CoreSettings
from llama_index.core.llms import LLM
from llama_index.core.indices.utils import truncate_text
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.prompts.default_prompt_selectors import DEFAULT_REFINE_PROMPT_SEL, DEFAULT_TEXT_QA_PROMPT_SEL, \
    DEFAULT_TREE_SUMMARIZE_PROMPT_SEL
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.response.utils import get_response_text
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.types import RESPONSE_TEXT_TYPE
from llama_index.core.prompts.prompt_utils import get_biggest_prompt
from llama_index.core.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT

# ======================== 配置日志 ========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('KGRAG')
logging.getLogger('KGRAG').setLevel(logging.INFO)
logging.getLogger('httpx').setLevel(logging.CRITICAL)
dispatcher = get_dispatcher(__name__)


# ======================== 辅助函数 ========================
def ngram_overlap(span, sent, n=3):
    """计算两个字符串之间的n-gram重叠度"""
    while (len(span) < n) or (len(sent) < n):
        n -= 1
    if n <= 0:
        return 0.0
    span = span.lower()
    sent = sent.lower()
    span_tokens = [token for token in span.split() if token not in string.punctuation]
    span_tokens = ''.join(span_tokens)
    sent_tokens = [token for token in sent.split() if token not in string.punctuation]
    sent_tokens = ''.join(sent_tokens)
    span_tokens = set([span_tokens[i:i + n] for i in range(len(span_tokens) - n + 1)])
    sent_tokens = set([sent_tokens[i:i + n] for i in range(len(sent_tokens) - n + 1)])
    overlap = span_tokens.intersection(sent_tokens)
    return float((len(overlap) + 0.01) / (len(span_tokens) + 0.01))


# ======================== 后处理器类 ========================
class NaivePostprocessor(BaseNodePostprocessor):
    """基础节点后处理器，用于排序和格式化节点"""
    dataset: str = Field

    @classmethod
    def class_name(cls) -> str:
        return "NaivePostprocessor"

    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        entity_order = {}
        sorted_nodes = []
        for i, node in enumerate(nodes):
            node_id = node.node.id_
            ent, seq_str = node_id.split('##')
            ctx_seq = int(seq_str)
            if ent not in entity_order:
                entity_order[ent] = len(entity_order)
            sorted_nodes.append((ent, ctx_seq, node))
        sorted_nodes.sort(key=lambda x: (entity_order[x[0]], x[1]))
        sorted_nodes = [node for _, _, node in sorted_nodes]

        prev_ent = ''
        for i in range(0, len(sorted_nodes)):
            temp_ent = sorted_nodes[i].node.id_.split('##')[0]
            if prev_ent == temp_ent:
                sorted_nodes[i].node.text = sorted_nodes[i].node.text[len(temp_ent + ': '):]
            prev_ent = temp_ent

        return sorted_nodes


class KGRetrievePostProcessor(BaseNodePostprocessor):
    """基于知识图谱的节点处理器，扩展相关实体"""
    dataset: str = Field
    ents: Set[str] = Field
    doc2kg: Dict[str, Dict[str, List[List[str]]]] = Field
    chunks_index: Dict[str, Dict[str, str]] = Field

    @classmethod
    def class_name(cls) -> str:
        return "KGRetrievePostprocessor"

    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        top_k = len(nodes)
        retrieved_ids = set()
        retrieved_ents = set()
        related_ents = set()
        highly_related_ents = set()
        textid2score = dict()
        ent_count = dict()
        ent_score = dict()

        # 处理初始检索到的节点
        for i, node in enumerate(nodes):
            node_id = node.node.id_
            retrieved_ids.add(node_id)
            textid2score[node_id] = node.score
            entity, seq_str = node_id.split('##')

            if i < (top_k // 2) and entity in retrieved_ents:
                highly_related_ents.add(entity)
            retrieved_ents.add(entity)

            if entity not in ent_count:
                ent_count[entity] = 0
                ent_score[entity] = 0.0
            ent_count[entity] += 1
            ent_score[entity] += node.score

        # 添加相关实体
        additional_ents = set()
        for node in nodes:
            node_id = node.node.id_
            entity, seq_str = node_id.split('##')
            if entity in self.doc2kg and seq_str in self.doc2kg[entity]:
                for triplet in self.doc2kg[entity][seq_str]:
                    h, r, t = triplet
                    if h in self.ents and h not in retrieved_ents:
                        additional_ents.add(h)
                    if t in self.ents and t not in retrieved_ents:
                        additional_ents.add(t)

        # 扩展实体关系
        hops = 1
        for hop in range(hops):
            related_ents = related_ents.union(additional_ents)
            temp_ents = set(additional_ents)
            additional_ents = set()
            for ent in temp_ents:
                if ent in self.doc2kg:
                    for idx_seq_str in self.doc2kg[ent]:
                        ctx_id = f'{ent}##{idx_seq_str}'
                        if ctx_id in retrieved_ids:
                            continue
                        for triplet in self.doc2kg[ent][idx_seq_str]:
                            h, r, t = triplet
                            if h in self.ents and h not in related_ents:
                                additional_ents.add(h)
                            if t in self.ents and t not in related_ents:
                                additional_ents.add(t)

        # 添加新节点
        additional_ids = set()
        avg_score = sum(node.score for node in nodes) / len(nodes) if nodes else 0
        for ent in (related_ents - retrieved_ents):
            if ent in self.chunks_index:
                for idx_seq_str in self.chunks_index[ent]:
                    ctx_id = f'{ent}##{idx_seq_str}'
                    if ctx_id not in retrieved_ids:
                        additional_ids.add(ctx_id)
                        textid2score[ctx_id] = ent_score.get(ent, avg_score) / (ent_count.get(ent, 1) + 1)

        added_nodes = []
        for ctx_id in additional_ids:
            ent, seq_str = ctx_id.split('##')
            if ent in self.chunks_index and seq_str in self.chunks_index[ent]:
                ctx_text = self.chunks_index[ent][seq_str]
                node = TextNode(id_=ctx_id, text=ctx_text)
                node = NodeWithScore(node=node, score=textid2score[ctx_id])
                added_nodes.append(node)

        return nodes + added_nodes


class GraphFilterPostProcessor(BaseNodePostprocessor):
    """基于知识图谱的节点过滤器，构建关系图并筛选节点"""
    dataset: str = Field
    topk: int = Field
    use_tpt: bool = Field
    ents: Set[str] = Field
    doc2kg: Dict[str, Dict[str, List[List[str]]]] = Field
    chunks_index: Dict[str, Dict[str, str]] = Field
    reranker: FlagReranker = Field

    @classmethod
    def class_name(cls) -> str:
        return "GraphFilterPostprocessor"

    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if not query_bundle:
            return nodes

        # 构建知识图谱
        g = nx.MultiGraph()
        ents = set()
        rels = set()

        # 收集实体和关系
        for node in nodes:
            ent, seq_str = node.node.id_.split('##')
            ents.add(ent)
            if ent in self.doc2kg and seq_str in self.doc2kg[ent]:
                for triplet in self.doc2kg[ent][seq_str]:
                    h, r, t = [x.strip() for x in triplet]
                    ents.add(h)
                    ents.add(t)
                    rels.add(r)
                    g.add_edge(h, t, rel=r, source=node.node.id_, weight=node.score)

        # 识别查询中提到的实体和关系
        mentioned_ents = {ent for ent in ents if ngram_overlap(ent, query_bundle.query_str) >= 0.90}
        mentioned_rels = {rel for rel in rels if ngram_overlap(rel, query_bundle.query_str) >= 0.90}

        # 扩展相关实体
        for node in nodes:
            ent, seq_str = node.node.id_.split('##')
            if ent in self.doc2kg and seq_str in self.doc2kg[ent]:
                for triplet in self.doc2kg[ent][seq_str]:
                    h, r, t = triplet
                    if h in mentioned_ents and r in mentioned_rels and t not in mentioned_ents:
                        mentioned_ents.add(t)
                    if t in mentioned_ents and r in mentioned_rels and h not in mentioned_ents:
                        mentioned_ents.add(h)

        # 构建连通子图
        wccs = list(nx.connected_components(g))
        sorted_wccs = sorted(wccs, key=len, reverse=True)
        cand_ctxs_lists = []

        # 生成候选上下文
        for wcc in sorted_wccs:
            cand_ctxs_list = []
            if len(wcc) > 1:
                subgraph = g.subgraph(wcc)
                mst = nx.maximum_spanning_tree(subgraph, weight='weight')
                cand_ctx_list = [edge[2]['source'] for edge in mst.edges(data=True) if edge[2]['source'] != 'query']
                if cand_ctx_list:
                    cand_ctxs_list.append(cand_ctx_list)
            else:
                sorted_edges = sorted(g.subgraph(wcc).edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
                for edge in sorted_edges:
                    if edge[2]['source'] != 'query':
                        cand_ctxs_list.append([edge[2]['source']])
                        break
            cand_ctxs_lists.extend(cand_ctxs_list)

        # 重新排序候选
        cand_strs = []
        for cand_ids_list in cand_ctxs_lists:
            ctx_str = ''
            for cand_id in cand_ids_list:
                ent, seq_str = cand_id.split('##')
                ctx_str += self.chunks_index[ent][seq_str]
            cand_strs.append(ctx_str)

        # 使用重排器排序
        if cand_strs:
            scores = self.reranker.compute_score([(query_bundle.query_str, cand_str) for cand_str in cand_strs])
            sorted_seqs = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
            wanted_ctxs = []
            for seq in sorted_seqs:
                if len(set(wanted_ctxs) | set(cand_ctxs_lists[seq])) > self.topk:
                    break
                wanted_ctxs.extend(cand_ctxs_lists[seq])
        else:
            wanted_ctxs = []

        # 返回最终节点
        return [node for node in nodes if node.node.id_ in wanted_ctxs]


# ======================== 响应合成器 ========================
class StructuredRefineResponse(BaseModel):
    """结构化响应模型"""
    answer: str = Field(description="基于上下文生成的答案")
    query_satisfied: bool = Field(description="查询是否被满足")


class DefaultRefineProgram:
    """默认响应生成程序"""

    def __init__(self, prompt: BasePromptTemplate, llm: LLM):
        self._prompt = prompt
        self._llm = llm

    def __call__(self, **kwds: Any) -> StructuredRefineResponse:
        answer = self._llm.predict(self._prompt, **kwds)
        return StructuredRefineResponse(answer=answer, query_satisfied=True)


class Refine(BaseSynthesizer):
    """响应精炼合成器"""

    def __init__(
            self,
            llm: Optional[LLM] = None,
            text_qa_template: Optional[BasePromptTemplate] = None,
            refine_template: Optional[BasePromptTemplate] = None,
            streaming: bool = False,
            verbose: bool = False,
    ):
        super().__init__(llm=llm, streaming=streaming)
        self._text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT_SEL
        self._refine_template = refine_template or DEFAULT_REFINE_PROMPT_SEL
        self._verbose = verbose

    def get_response(
            self,
            query_str: str,
            text_chunks: Sequence[str],
            prev_response: Optional[RESPONSE_TEXT_TYPE] = None,
            **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        response = None
        for text_chunk in text_chunks:
            if prev_response is None:
                response = self._give_response_single(query_str, text_chunk, **response_kwargs)
            else:
                response = self._refine_response_single(prev_response, query_str, text_chunk, **response_kwargs)
            prev_response = response
        return response or "Empty Response"

    def _give_response_single(
            self,
            query_str: str,
            text_chunk: str,
            **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        program = DefaultRefineProgram(text_qa_template, self._llm)
        return program(context_str=text_chunk, **response_kwargs).answer


class CompactAndRefine(Refine):
    """紧凑响应合成器"""

    def get_response(
            self,
            query_str: str,
            text_chunks: Sequence[str],
            prev_response: Optional[RESPONSE_TEXT_TYPE] = None,
            **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        return super().get_response(
            query_str=query_str,
            text_chunks=text_chunks,
            prev_response=prev_response,
            **response_kwargs,
        )


def get_response_synthesizer(
        llm: Optional[LLM] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        refine_template: Optional[BasePromptTemplate] = None,
        response_mode: ResponseMode = ResponseMode.COMPACT,
        streaming: bool = False,
) -> BaseSynthesizer:
    """获取响应合成器"""
    return CompactAndRefine(
        llm=llm,
        text_qa_template=text_qa_template,
        refine_template=refine_template,
        streaming=streaming,
    )


# ======================== 主程序功能 ========================
class TokenCounter:
    """令牌计数器"""

    def __init__(self, ratio=4):
        self.total_input_chars = 0
        self.total_output_chars = 0
        self.ratio = ratio

    def count_input(self, text: str):
        self.total_input_chars += len(text)

    def count_output(self, text: str):
        self.total_output_chars += len(text)

    @property
    def input_tokens(self):
        return int(self.total_input_chars / self.ratio)

    @property
    def output_tokens(self):
        return int(self.total_output_chars / self.ratio)

    def save_to_file(self, filepath):
        stats = {
            "input_chars": self.total_input_chars,
            "output_chars": self.total_output_chars,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens
        }
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)


def read_data(args):
    """读取数据"""
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f'{args.data_path} not found')

    if args.dataset == 'hotpotqa':
        with open(args.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif args.dataset == 'musique':
        data = []
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')


def read_kg(args, data):
    """读取知识图谱"""
    doc2kg = {}
    if args.dataset == 'hotpotqa':
        ents = set()
        for sample in data:
            for ctx in sample['context']:
                ents.add(ctx[0])
        for ent in tqdm(ents, desc='Loading KGs'):
            subkg_path = os.path.join(args.kg_dir, f'{ent.replace("/", "_")}.json')
            if os.path.exists(subkg_path):
                with open(subkg_path, 'r', encoding='utf-8') as f:
                    subkg = json.load(f)
                    repkg = {k: v for k, v in subkg.items() if v}
                    if repkg:
                        doc2kg[ent] = repkg
    elif args.dataset == 'musique':
        kg_path = os.path.join(args.kg_dir, 'musique_kg.json')
        if os.path.exists(kg_path):
            with open(kg_path, 'r', encoding='utf-8') as f:
                doc2kg = json.load(f)
    logger.info(f'Loaded kg for {len(doc2kg.keys())} entities from {args.dataset}')
    return doc2kg


def write_prediction(args, data, prediction):
    """写入预测结果"""
    if args.dataset == 'hotpotqa':
        with open(args.result_path, 'w', encoding='utf-8') as f:
            json.dump(prediction, f)
    elif args.dataset == 'musique':
        with open(args.result_path, 'w', encoding='utf-8') as f:
            for sample in data:
                sample_id = sample['id']
                sample['predicted_answer'] = prediction['answer'][sample_id]
                sample['predicted_support_idxs'] = prediction['sp'][sample_id]
                sample['predicted_answerable'] = sample['answerable']
                f.write(json.dumps(sample) + '\n')
    logger.info(f'Prediction written to {args.result_path}')


def process_sample(args, sample, kg):
    """处理单个样本"""
    sample_id = sample['_id'] if args.dataset == 'hotpotqa' else sample['id']
    sample_question = sample['question']
    ents = set()
    subkg = {}
    doc_chunks = []
    chunks_index = {}

    try:
        # 准备数据
        if args.dataset == 'hotpotqa':
            ctxs = sample['context']
            ents = {ctx[0] for ctx in ctxs}
            for ctx in ctxs:
                ent = ctx[0]
                chunks_index[ent] = {}
                for i, text in enumerate(ctx[1]):
                    if ent in kg and str(i) in kg[ent] and kg[ent][str(i)]:
                        subkg.setdefault(ent, {})[str(i)] = kg[ent][str(i)]
                    doc_chunk = TextNode(text=f'{ent}: {text}', id_=f'{ent}##{i}')
                    doc_chunks.append(doc_chunk)
                    chunks_index[ent][str(i)] = f'{ent}: {text}'

        elif args.dataset == 'musique':
            ctxs = sample['paragraphs']
            for ctx in ctxs:
                ent = ctx['title']
                idx = ctx['idx']
                seq = ctx['seq']
                ents.add(ent)
                chunks_index.setdefault(ent, {})
                key = f'{idx}##{seq}'
                if ent in kg and str(seq) in kg[ent] and kg[ent][str(seq)]:
                    subkg.setdefault(ent, {})[key] = kg[ent][str(seq)]
                doc_chunk = TextNode(text=f'{ent}: {ctx["paragraph_text"]}', id_=f'{idx}##{ent}##{seq}')
                doc_chunks.append(doc_chunk)
                chunks_index[ent][key] = f'{ent}: {ctx["paragraph_text"]}'

        # 构建索引和查询引擎
        index = VectorStoreIndex(doc_chunks)
        retriever = VectorIndexRetriever(index=index, similarity_top_k=args.top_k)

        qa_rag_template_str = (
            "Context information is below.\n{context_str}\n"
            "Think step by step but give a short factoid answer (as few words as possible) "
            "based on the context and your own knowledge.\n"
            "Q: {query_str}\nA: "
        )
        qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)

        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            text_qa_template=qa_rag_prompt_template
        )

        # 设置后处理器
        expansion_pp = KGRetrievePostProcessor(
            dataset=args.dataset,
            ents=ents,
            doc2kg=subkg,
            chunks_index=chunks_index
        )

        bge_reranker = FlagReranker(model_name_or_path=args.reranker, device='cuda')

        filter_pp = GraphFilterPostProcessor(
            dataset=args.dataset,
            use_tpt=args.use_tpt,
            topk=args.top_k,
            ents=ents,
            doc2kg=subkg,
            chunks_index=chunks_index,
            reranker=bge_reranker
        )

        naive_pp = NaivePostprocessor(dataset=args.dataset)

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[expansion_pp, filter_pp, naive_pp]
        )

        # 执行查询
        response = query_engine.query(sample_question)
        prediction = response.response

        # 处理支持证据
        if args.dataset == 'hotpotqa':
            sps = []
            for source_node in response.source_nodes:
                parts = source_node.node.id_.split('##')
                if len(parts) >= 2:
                    ent, seq_str = parts[:2]
                    try:
                        seq = int(seq_str)
                        if seq >= 0:
                            sps.append([ent, seq])
                    except ValueError:
                        continue
        elif args.dataset == 'musique':
            sps = []
            for source_node in response.source_nodes:
                parts = source_node.node.id_.split('##')
                if len(parts) >= 1:
                    try:
                        idx = int(parts[0])
                        if idx >= 0:
                            sps.append(idx)
                    except ValueError:
                        continue

        # 统计token
        context_texts = [node.text for node in response.source_nodes]
        full_input = sample_question + "\n".join(context_texts)
        Settings.token_counter.count_input(full_input)
        Settings.token_counter.count_output(prediction)

        return sample_id, prediction, sps

    except Exception as e:
        logger.error(f'Sample {sample_id}, Error: {e}')
        return sample_id, '', []


def kgrag_distractor_predict(args, data, kg):
    """执行预测"""
    prediction = {'answer': {}, 'sp': {}}
    sps_count = 0
    for sample in tqdm(data, desc='Processing samples'):
        sample_id, sample_prediction, sample_sps = process_sample(args, sample, kg)
        prediction['answer'][sample_id] = sample_prediction
        prediction['sp'][sample_id] = sample_sps
        sps_count += len(sample_sps)
    logger.info(f'Average number of supporting facts: {sps_count / len(data):.2f}')
    return prediction


def init_model(args):
    """初始化模型"""
    Settings.llm = Ollama(model=args.model_name, request_timeout=200)
    Settings.embed_model = OllamaEmbedding(model_name=args.embed_model_name)
    Settings.token_counter = TokenCounter()


def main(args):
    """主函数"""
    data = read_data(args)
    init_model(args)
    kg = read_kg(args, data)
    prediction = kgrag_distractor_predict(args, data, kg)
    write_prediction(args, data, prediction)
    Settings.token_counter.save_to_file(args.token_output_path)
    logger.info(f"Token统计已保存至: {args.token_output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hotpotqa', help='Dataset name')
    parser.add_argument('--data_path', type=str, default='../data/hotpotqa/hotpot_dev_distractor_v1.json',
                        help='Path to data file')
    parser.add_argument('--kg_dir', type=str, default='../data/hotpotqa/kgs/extract_subkgs', help='KG directory')
    parser.add_argument('--use_tpt', type=bool, default=False, help='Use triplet representation')
    parser.add_argument('--result_path', type=str, default='../output/hotpot/hotpot_dev_distractor_v1_kgrag.json',
                        help='Result file path')
    parser.add_argument('--embed_model_name', type=str, default='mxbai-embed-large:latest', help='Embedding model')
    parser.add_argument('--model_name', type=str, default='llama3:8b', help='Ollama model')
    parser.add_argument('--reranker', type=str, default='../model/bge-reranker-large', help='Reranker model path')
    parser.add_argument('--top_k', type=int, default=10, help='Top k similar documents')
    parser.add_argument('--token_output_path', type=str, default='./tokenout-MST+BFS2.json', help='Token stats path')
    args = parser.parse_args()

    main(args)