import os
import json
import argparse
import string
import re
import time
import networkx as nx
import logging
from tqdm import tqdm
from typing import List, Dict, Optional, Set, Sequence, Any, Callable, Generator, Type, cast, AsyncGenerator
from FlagEmbedding import FlagReranker
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate, StorageContext, load_index_from_storage
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
from llama_index.core.prompts.default_prompt_selectors import DEFAULT_REFINE_PROMPT_SEL, DEFAULT_TEXT_QA_PROMPT_SEL
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.response.utils import get_response_text
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.types import RESPONSE_TEXT_TYPE
from llama_index.core.prompts.prompt_utils import get_biggest_prompt
from llama_index.core.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT
from llama_index.core.indices.prompt_helper import PromptHelper

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
    span_tokens = set(span_tokens[i:i + n] for i in range(len(span_tokens) - n + 1))
    sent_tokens = set(sent_tokens[i:i + n] for i in range(len(sent_tokens) - n + 1))
    overlap = span_tokens.intersection(sent_tokens)
    return float((len(overlap) + 0.01) / (len(span_tokens) + 0.01))

    # ======================== 后处理器类 ========================


class NaivePostprocessor(BaseNodePostprocessor):
    """基础节点后处理器，用于排序和格式化节点"""
    dataset: str = Field(default="hotpotqa")

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
        for node in nodes:
            node_id = node.node.id_
            parts = node_id.split('##')
            if len(parts) < 2:
                continue
            ent = parts[0]
            seq_str = parts[1]
            try:
                ctx_seq = int(seq_str)
            except ValueError:
                continue
            if ent not in entity_order:
                entity_order[ent] = len(entity_order)
            sorted_nodes.append((ent, ctx_seq, node))
        sorted_nodes.sort(key=lambda x: (entity_order[x[0]], x[1]))
        sorted_nodes = [node for _, _, node in sorted_nodes]

        prev_ent = ''
        for i in range(len(sorted_nodes)):
            node_id = sorted_nodes[i].node.id_
            parts = node_id.split('##')
            if len(parts) < 2:
                continue
            temp_ent = parts[0]
            if prev_ent == temp_ent:
                sorted_nodes[i].node.text = sorted_nodes[i].node.text[len(temp_ent + ': '):]
            prev_ent = temp_ent

        return sorted_nodes


class KGRetrievePostProcessor(BaseNodePostprocessor):
    """基于知识图谱的节点处理器，扩展相关实体"""
    dataset: str = Field(default="hotpotqa")
    ents: Set[str] = Field()
    doc2kg: Dict[str, Dict[str, List[List[str]]]] = Field()
    chunks_index: Dict[str, Dict[str, str]] = Field()

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
        textid2score = {}
        ent_count = {}
        ent_score = {}

        # 处理初始检索到的节点
        for i, node in enumerate(nodes):
            node_id = node.node.id_
            retrieved_ids.add(node_id)
            textid2score[node_id] = node.score

            parts = node_id.split('##')
            if len(parts) < 2:
                continue
            entity = parts[0]

            if i < (top_k // 2) and entity in retrieved_ents:
                pass
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
            parts = node_id.split('##')
            if len(parts) < 2:
                continue
            entity = parts[0]
            seq_str = parts[1]

            if entity not in self.doc2kg or seq_str not in self.doc2kg[entity]:
                continue

            for triplet in self.doc2kg[entity][seq_str]:
                h, r, t = triplet
                if h in self.ents and h not in retrieved_ents:
                    additional_ents.add(h)
                if t in self.ents and t not in retrieved_ents:
                    additional_ents.add(t)

        # 添加新节点
        additional_ids = set()
        avg_score = sum(node.score for node in nodes) / len(nodes) if nodes else 0
        for ent in additional_ents:
            if ent not in self.chunks_index or not self.chunks_index[ent]:
                continue

            for seq_str in self.chunks_index[ent]:
                ctx_id = f'{ent}##{seq_str}'
                if ctx_id in retrieved_ids:
                    continue

                additional_ids.add(ctx_id)
                score = ent_score.get(ent, avg_score) / (ent_count.get(ent, 1) + 1)
                textid2score[ctx_id] = score

        added_nodes = []
        for ctx_id in additional_ids:
            parts = ctx_id.split('##')
            if len(parts) < 2:
                continue
            ent = parts[0]
            seq_str = parts[1]

            if ent in self.chunks_index and seq_str in self.chunks_index[ent]:
                ctx_text = self.chunks_index[ent][seq_str]
                node = TextNode(id_=ctx_id, text=ctx_text)
                node = NodeWithScore(node=node, score=textid2score[ctx_id])
                added_nodes.append(node)

        return nodes + added_nodes


class GraphFilterPostProcessor(BaseNodePostprocessor):
    """基于知识图谱的节点过滤器，构建关系图并筛选节点"""
    dataset: str = Field(default="hotpotqa")
    topk: int = Field()
    use_tpt: bool = Field(default=False)
    ents: Set[str] = Field()
    doc2kg: Dict[str, Dict[str, List[List[str]]]] = Field()
    chunks_index: Dict[str, Dict[str, str]] = Field()
    reranker: FlagReranker = Field()

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
            node_id = node.node.id_
            parts = node_id.split('##')
            if len(parts) < 2:
                continue
            ent = parts[0]
            seq_str = parts[1]
            ents.add(ent)

            if ent in self.doc2kg and seq_str in self.doc2kg[ent]:
                for triplet in self.doc2kg[ent][seq_str]:
                    h, r, t = [x.strip() for x in triplet]
                    ents.add(h)
                    ents.add(t)
                    rels.add(r)
                    g.add_edge(h, t, rel=r, source=node_id, weight=node.score)

        # 识别查询中提到的实体和关系
        query_str = query_bundle.query_str
        mentioned_ents = {ent for ent in ents if ngram_overlap(ent, query_str) >= 0.90}
        mentioned_rels = {rel for rel in rels if ngram_overlap(rel, query_str) >= 0.90}

        # 扩展相关实体
        for node in nodes:
            node_id = node.node.id_
            parts = node_id.split('##')
            if len(parts) < 2:
                continue
            ent = parts[0]
            seq_str = parts[1]

            if ent not in self.doc2kg or seq_str not in self.doc2kg[ent]:
                continue

            for triplet in self.doc2kg[ent][seq_str]:
                h, r, t = triplet
                if h in mentioned_ents and r in mentioned_rels and t not in mentioned_ents:
                    mentioned_ents.add(t)
                if t in mentioned_ents and r in mentioned_rels and h not in mentioned_ents:
                    mentioned_ents.add(h)

        # 构建连通子图
        wccs = list(nx.connected_components(g))
        sorted_wccs = sorted(wccs, key=len, reverse=True)
        cand_ids_lists = []

        # 生成候选上下文
        for wcc in sorted_wccs:
            if len(wcc) > 1:
                subgraph = g.subgraph(wcc)
                try:
                    mst = nx.maximum_spanning_tree(subgraph, weight='weight')
                    cand_list = [edge[2]['source'] for edge in mst.edges(data=True)
                                 if edge[2]['source'] != 'query' and 'source' in edge[2]]
                    if cand_list:
                        cand_ids_lists.append(cand_list)
                except:
                    pass
            else:
                sorted_edges = sorted(g.subgraph(wcc).edges(data=True), key=lambda x: x[2].get('weight', 0),
                                      reverse=True)
                for edge in sorted_edges:
                    if 'source' in edge[2] and edge[2]['source'] != 'query':
                        cand_ids_lists.append([edge[2]['source']])
                        break

        # 重新排序候选
        cand_strs = []
        for cand_ids in cand_ids_lists:
            ctx_str = ''
            for cand_id in cand_ids:
                parts = cand_id.split('##')
                if len(parts) < 2:
                    continue
                ent = parts[0]
                seq_str = parts[1]
                if ent in self.chunks_index and seq_str in self.chunks_index[ent]:
                    ctx_str += self.chunks_index[ent][seq_str] + " "
            cand_strs.append(ctx_str.strip())

        # 使用重排器排序
        if cand_strs:
            try:
                scores = self.reranker.compute_score([(query_str, cand_str) for cand_str in cand_strs])
                sorted_seqs = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
                wanted_ctxs = []
                for seq in sorted_seqs:
                    if len(set(wanted_ctxs) | set(cand_ids_lists[seq])) > self.topk:
                        break
                    wanted_ctxs.extend(cand_ids_lists[seq])
            except:
                wanted_ctxs = []
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


# ======================== KG-RAG 主逻辑 ========================
def kg_rag_predict(data, doc2kg, top_k=5, persist_dir=None, reranker='../model/bge-reranker-large'):
    """执行KG-RAG预测"""
    prediction = {'answer': {}, 'sp': {}}

    # 准备文档块和索引
    doc_chunks = []
    chunks_index = {}
    ents = set()
    for sample in data:
        for ctx in sample['context']:
            ent = ctx[0]
            ents.add(ent)
            if ent not in chunks_index:
                chunks_index[ent] = {}
            for i, text in enumerate(ctx[1]):
                doc_chunk = TextNode(text=f'{ent}: {text}', id_=f'{ent}##{i}')
                doc_chunks.append(doc_chunk)
                chunks_index[ent][str(i)] = doc_chunk.text

    # 加载或创建索引
    if persist_dir and os.path.exists(persist_dir):
        logger.info('加载持久化索引')
        sc = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(sc)
    else:
        logger.info('创建新索引')
        index = VectorStoreIndex(doc_chunks, show_progress=True)
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=persist_dir)

    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    # 设置响应合成器
    qa_template = (
        "Context information is below.\n{context_str}\n"
        "Give a short factoid answer (as few words as possible).\n"
        "Q: {query_str}\nA: "
    )
    qa_prompt = PromptTemplate(qa_template)
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        text_qa_template=qa_prompt
    )

    # 设置后处理器
    kg_expander = KGRetrievePostProcessor(
        ents=ents,
        doc2kg=doc2kg,
        chunks_index=chunks_index
    )

    bge_reranker = FlagReranker(model_name_or_path=reranker)
    kg_filter = GraphFilterPostProcessor(
        topk=top_k,
        ents=ents,
        doc2kg=doc2kg,
        chunks_index=chunks_index,
        reranker=bge_reranker
    )

    naive_processor = NaivePostprocessor()

    # 创建查询引擎
    engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[kg_expander, kg_filter, naive_processor]
    )

    # 处理所有样本
    sps_count = []
    for sample in tqdm(data, desc="处理样本"):
        sample_id = sample['_id']
        question = sample['question']
        answer = sample['answer']

        try:
            response = engine.query(question)
            pred_answer = response.response
            sps = []
            for source_node in response.source_nodes:
                node_id = source_node.node.id_
                parts = node_id.split('##')
                if len(parts) >= 2:
                    ent, seq_str = parts[:2]
                    try:
                        seq = int(seq_str)
                        sps.append([ent, seq])
                    except ValueError:
                        pass

            prediction['answer'][sample_id] = pred_answer
            prediction['sp'][sample_id] = sps
            sps_count.append(len(sps))
        except Exception as e:
            logger.error(f"样本 {sample_id} 处理失败: {str(e)}")
            prediction['answer'][sample_id] = ""
            prediction['sp'][sample_id] = []

    avg_sps = sum(sps_count) / len(sps_count) if sps_count else 0
    logger.info(f'平均支持事实数量: {avg_sps:.2f}')
    return prediction


def load_kg_data(kg_dir, ents):
    """加载知识图谱数据"""
    doc2kg = {}
    logger.info(f"从目录加载知识图谱: {kg_dir}")
    for ent in tqdm(ents, desc="加载实体知识图谱"):
        safe_ent = ent.replace("/", "_")
        kg_path = os.path.join(kg_dir, f'{safe_ent}.json')
        if not os.path.exists(kg_path):
            continue

        with open(kg_path, 'r', encoding='utf-8') as f:
            try:
                kg_data = json.load(f)
                # 清理空条目
                kg_data = {k: v for k, v in kg_data.items() if v}
                if kg_data:
                    doc2kg[ent] = kg_data
            except json.JSONDecodeError:
                logger.warning(f"无效的JSON文件: {kg_path}")

    logger.info(f"加载了 {len(doc2kg)} 个实体的知识图谱")
    return doc2kg


def main(args):
    # 加载数据
    logger.info(f"从 {args.data_path} 加载数据")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 收集所有实体
    ents = set()
    for sample in data:
        for ctx in sample['context']:
            ents.add(ctx[0])
    logger.info(f"共发现 {len(ents)} 个唯一实体")

    # 加载知识图谱
    doc2kg = load_kg_data(args.kg_dir, ents)

    # 初始化模型
    logger.info(f"初始化Ollama模型: {args.model_name}")
    Settings.llm = Ollama(model=args.model_name, request_timeout=200)
    logger.info(f"初始化嵌入模型: {args.embed_model_name}")
    Settings.embed_model = OllamaEmbedding(model_name=args.embed_model_name)

    # 执行预测
    prediction = kg_rag_predict(
        data,
        doc2kg,
        top_k=args.top_k,
        persist_dir=args.persist_dir,
        reranker=args.reranker
    )

    # 保存结果
    logger.info(f"将结果保存到 {args.result_path}")
    with open(args.result_path, 'w', encoding='utf-8') as f:
        json.dump(prediction, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KG-RAG 知识图谱增强检索系统')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='hotpotqa', help='数据集名称')
    parser.add_argument('--data_path', type=str, required=True, help='数据文件路径')
    parser.add_argument('--kg_dir', type=str, required=True, help='知识图谱目录')
    parser.add_argument('--result_path', type=str, required=True, help='结果文件路径')
    parser.add_argument('--persist_dir', type=str, default='./storage', help='索引存储目录')

    # 模型参数
    parser.add_argument('--model_name', type=str, default='llama3:8b', help='Ollama模型名称')
    parser.add_argument('--embed_model_name', type=str, default='mxbai-embed-large', help='嵌入模型名称')
    parser.add_argument('--reranker', type=str, default='../model/bge-reranker-large', help='重排模型路径')
    parser.add_argument('--top_k', type=int, default=10, help='检索top-k文档')

    args = parser.parse_args()

    # 确保存储目录存在
    os.makedirs(args.persist_dir, exist_ok=True)

    main(args)