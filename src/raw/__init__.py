#!/usr/bin/enviroments python
# -*- coding:utf-8 _*-
"""
@File:   __init__.py
@IDE:    Pycharm
@Des:
"""

import os
import random
import sys

import numpy.random
import torch.nn
from torch import Tensor

import fairseq
from contextlib import contextmanager

from src.config import BaseModelConfig
from src.config.dag_baseline import DAGBaseModelConfig
from src.models import BaseModel
from src.utils.log import logger

fairseq_path = os.path.dirname(os.path.dirname(fairseq.__file__))
sys.path.insert(1, f"{fairseq_path}")
from fs_plugins.models.glat_decomposed_with_link import GlatDecomposedLink
from fs_plugins.models.hub_interface import DATHubInterface


@contextmanager
def torch_seed(seed):
    # modified from lunanlp
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        state_cuda = torch.cuda.random.get_rng_state()
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        if torch.cuda.is_available():
            torch.cuda.random.set_rng_state(state_cuda)


class Baseline(BaseModel):
    conf: DAGBaseModelConfig
    da_transformer: GlatDecomposedLink

    def __init__(self, config: DAGBaseModelConfig, hub_model: DATHubInterface):
        super().__init__(config)
        self.hub_task = None
        self.hub_cfg = None
        self.conf = config
        logger.info("Initiating DAG Baseline Model...")
        self.init_generative_model(hub_model)

    def init_hub_model(self) -> DATHubInterface:

        import concurrent.futures

        try:
            # import fairseq
            # fairseq_path = os.path.dirname(os.path.dirname(fairseq.__file__))
            # sys.path.insert(1, f"{fairseq_path}")
            # from fs_plugins.models.glat_decomposed_with_link import GlatDecomposedLink
            # Now load model
            model = GlatDecomposedLink.from_pretrained(**self.conf.dat_model_args)
        except ModuleNotFoundError as e:
            logger.critical(f"在尝试加载DA-Transformer的时候无法加载依赖包，具体报错内容如下：\n{e}")
            exit(1)
        except KeyError as e:
            logger.critical(f"在尝试初始化DA-Transformer中出现参数缺失，具体内容如下：\n{e}")
            exit(1)
        except concurrent.futures.process.BrokenProcessPool as e:
            logger.critical(f"初始化DA-Transformer中出现了错误，具体内容如下：\n{e}")
            exit(1)
        return model

    def init_generative_model(self, hub_model):

        logger.info("Initiating DA-Transformer ...")
        # hub_model = self.init_hub_model()
        self.hub_cfg = hub_model.cfg
        self.hub_task = hub_model.task
        self.da_transformer = hub_model.model
        logger.info("DA-Transformer initiated.")

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, net_input=None, **kwargs):

        encoder_out = self.da_transformer.encoder(src_tokens, src_lengths=src_lengths)
        # length prediction
        length_out = self.da_transformer.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.da_transformer.decoder.forward_length_prediction(
            length_out=length_out, encoder_out=encoder_out, tgt_tokens=tgt_tokens
        )
        rand_seed = numpy.random.randint(0, 19260817)

        # decode
        word_ins_out, links = self.extract_features(
            prev_output_tokens, encoder_out, net_input, rand_seed, require_links=True
        )

        ret = {
            "word_ins": {
                "out": word_ins_out,  # [batch_size, max_predict_length_in_batch ,vocab_size]
                "tgt": tgt_tokens,  # [batch_size, max_tgt_length_in_batch]
                "mask": tgt_tokens.ne(self.hub_task.target_dictionary.pad()),
                # [batch_size, max_tgt_length_in_batch ,tgt_length]
                "nll_loss": True,
            },
            'links': links,
            'length': {
                "out": length_out,  # [batch_size, 256]
                "tgt": length_tgt,  # [batch_size] , stores max_tgt_length_in_batch
                "factor": self.da_transformer.decoder.length_loss_factor,
            }
        }
        encoder_out_flag = False
        for each_key in encoder_out:
            if isinstance(encoder_out[each_key], torch.Tensor):
                if encoder_out[each_key].isnan().any():
                    logger.critical(f"encoder_out中{each_key}出现NAN")
                    encoder_out_flag = True
            elif isinstance(encoder_out[each_key], list):
                for each in encoder_out[each_key]:
                    if each.isnan().any():
                        logger.critical(f"encoder_out中{each_key}出现NAN")
                        encoder_out_flag = True
        if encoder_out_flag:
            exit(1)
        if length_out.isnan().any():
            logger.critical(f"length put 有nan")
            exit(1)
        if length_tgt.isnan().any():
            logger.critical(f"length_tgt 有nan")
            exit(1)
        if word_ins_out.isnan().any():
            logger.critical("word ins out 有NAN")
            exit(1)
        if links.isnan().any():
            logger.critical("links 有NAN")
            exit(1)

        return ret

    def extract_features(self, prev_output_tokens, encoder_out, net_input, rand_seed, require_links=False,
                         training=True):
        with torch_seed(rand_seed):
            features, _ = self.da_transformer.decoder.extract_features(
                prev_output_tokens,
                net_input,
                encoder_out=encoder_out,
                embedding_copy=False
            )
            # 这里主要是看一下是不是features会拿到对应的NAN
            if features.isnan().any():
                logger.critical(f"features存在NAN")
            # word_ins_out = self.decoder.output_layer(features)
            word_ins_out = self.da_transformer.decoder.output_projection(features)

            links = None
            if require_links:
                links = self.da_transformer.extract_links(features, \
                                           prev_output_tokens, \
                                           self.da_transformer.decoder.link_positional, \
                                           self.da_transformer.decoder.query_linear, \
                                           self.da_transformer.decoder.key_linear, \
                                           self.da_transformer.decoder.gate_linear,
                                           net_input=net_input,
                                           training=training
                                           )

        return word_ins_out, links

    def forward_decoder(self, decoder_out, encoder_out, **kwargs):
        """

        Args:
            decoder_out:
            encoder_out:
            **kwargs:

        Returns:

        """
        output_tokens = decoder_out.output_tokens
        rand_seed = random.randint(0, 19260817)
        bsz, seqlen = output_tokens.shape
        prev_output_tokens_position = (
                torch.arange(seqlen, dtype=torch.long, device=output_tokens.device).unsqueeze(0).expand(bsz,
                                                                                                        -1) + 1). \
            masked_fill(output_tokens == self.da_transformer.tgt_dict.pad_index, 0)
        prev_output_tokens_segid = prev_output_tokens_position.masked_fill(
            output_tokens != self.da_transformer.tgt_dict.pad_index, 1). \
            masked_fill_(output_tokens == self.da_transformer.tgt_dict.eos_index, 2)
        output_length = (output_tokens != self.da_transformer.tgt_dict.pad_index).sum(dim=-1) - 1

        bound_end = torch.zeros_like(prev_output_tokens_segid).masked_fill_(
            output_tokens != self.da_transformer.tgt_dict.pad_index,
            1) * output_length.unsqueeze(-1)

        net_input = {
            "prev_output_tokens_segid": prev_output_tokens_segid,
            "bound_end": bound_end
        }

        output_logits, links = self.da_transformer.extract_features(output_tokens, encoder_out, net_input, rand_seed,
                                                                    require_links=True, training=False)

        links = self.da_transformer.restore_valid_links(links)
        return output_logits, links

    def generate_index(self, src_tokens, src_length, **kwargs):
        """

        Args:
            src_tokens:
            src_length:
            **kwargs:

        Returns:

        """
        encoder_out = self.da_transformer.forward_encoder([src_tokens, src_length])
        # logger.debug(f"src_length is {src_length}")
        prev_decoder_out = self.da_transformer.initialize_output_tokens(encoder_out, src_tokens)

        prev_decoder_out = prev_decoder_out._replace(
            step=0,
            max_step=0
        )
        word_ins_out, links = self.forward_decoder(prev_decoder_out, encoder_out, decoding_graph=True)
        return word_ins_out, links

    def reformat_for_save(self, ret):
        """

        Args:
            ret:

        Returns:
            {  "word_graph":Tensor[?, vocab_size],
               "link_graph": Tensor[?, ?-1]
            }
        """
        word_graph = ret['word_ins']['out']
        links_graph = self.da_transformer.restore_valid_links(ret['links'])
        return {
            'word_graph': word_graph,
            'link_graph': links_graph
        }


class BaseRetriever(BaseModel):

    def __init__(self, config: BaseModelConfig):
        super().__init__(config)
        self.length_penalty_factor = 1.0
        self.length_penalty_alpha = 1.0

    def forward(self, word_ins_out: Tensor, tgt_tokens: Tensor, links: Tensor):
        """
        把原始版本的forward方法拆分出来，方便后续的继承和修改
        Args:
            word_ins_out:
            tgt_tokens:
            links:

        Returns:

        """
        return self.raw_forward(word_ins_out, tgt_tokens, links)

    def raw_forward(self, word_ins_out: Tensor, tgt_tokens: Tensor, links: Tensor):
        """
        compute the tgt_tokens max generate prob when given graph with nodes as word_ins_out and edges as links
        Args:
            word_ins_out: [batch_size, max_predict_length_in_batch, vocab_size]
            tgt_tokens: [batch_size, max_tgt_length_in_batch]
            links: [batch_size, max_predict_length_in_batch, max_predict_length_in_batch]

        Returns:
            max prob in graph when given tgt
            [batch_size, 1]
        """
        # links 中最后一行？列？是nan，需要转为-inf
        # links = torch.where(torch.isnan(links), torch.tensor(float("-inf"), device=self.device), links)
        # 维护一个张量prob [batch_size, node_nums, tgt_length]
        # 表示为对于图中第node_index个节点，当前节点是生成第tgt_index个token的最大概率是多少
        # 通过这种方法我们就可以在生成计算表后查询每一个node中tgt_length-1位置的生成概率中的max即为所需要的值
        batch_size, node_num, tgt_num = word_ins_out.shape[0], word_ins_out.shape[1], tgt_tokens.shape[1]
        prob = torch.full([batch_size, node_num, tgt_num], float('-inf'),
                          device=self.device)  # [batch_size, node_num, tgt_num]
        # 那么在前边第i_node行中，位置在i_node及之后的的tgt是不可能生成的，所以这个位置的概率应该是0,也就是 -inf
        # 第一行肯定是<s>
        prob[:, 0, 0] = 0

        # 然后开始个token进行计算
        # 对于每一个token,需要去计算每一个node生成的是当前token的概率
        # 没有必要去计算太前（前边的token还没有开始生成）或者是太后（后续其他token也需要生成）
        # 当前位置的最大概率应该是前边所有node中tgt_index-1的生成概率*转移概率*当前位置对应token的生成概率，求max

        for tgt_index in range(1, tgt_num):
            # [batch_size, 1, 1] 这里是找到当前每一句话中的当前tgt_token，并且把形状修改了
            selected_token = tgt_tokens[:, tgt_index, None, None]
            # [batch_size, node_num, 1] 保持头尾维度不变，中间用传播方法直接复制node_num个，这里是为了后续能够把每一个node中的tgt都拿到
            selected_token = selected_token.expand(-1, node_num, -1)
            # [batch_size, 1, node_num] 从word ins out 找出每一个node的当前token生成概率,因为考虑到后边，这里需要固定每一列不变
            selected_prob = word_ins_out.gather(dim=2, index=selected_token).transpose(1, 2)
            # [batch_size, node_num, node_num] 生成概率+转移概率+之前的概率,

            # gen_prob = links + prob[:, :, tgt_index - 1].unsqueeze(2) + selected_prob
            gen_prob = prob[:, :, tgt_index - 1].unsqueeze(2) + selected_prob
            # [batch_size, node_num]
            prob[:, :, tgt_index], _ = torch.max(gen_prob, dim=-1)  # 该考虑到pad的问题了
        # 查看tgt序列中pad，也就是第一个2的位置，代表句子的结束，这里是包含了</s>的了
        # 在(tgt_tokens == 2).int().argmax(-1)之后获得是[batch_size]的形状,需要扩展为[batch_size, 1]形状，
        # 然后进而传播到[batch_size, node_num]形状,然后进一步修改为[batch_size, node_num，1]
        end_positions = (tgt_tokens == 2).int().argmax(-1, keepdim=True).expand(-1, node_num).unsqueeze(-1)
        try:
            # [batch_size, node_num, 1] 从prob中找到每一个node中</s>位置的生成概率
            selected_probs = torch.gather(prob, dim=2, index=end_positions)
        except IndexError as e:
            logger.critical(f"error:{e}")
            logger.debug(
                f"end_positions shape: {end_positions.shape}"
                f"tgt_token_size: {tgt_tokens.shape}")
            logger.debug(end_positions)
            exit(1)
        # 最大概率和对应的index, [batch_size,1]
        scores, _ = torch.max(selected_probs, dim=1)
        return scores

    def length_penalty(self, prob: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """
        引入Google的长度惩罚
        Args:
            prob:
            length:

        Returns:

        """
        penalty = ((5 + length) ** self.length_penalty_alpha) / (
                (5 + self.length_penalty_factor) ** self.length_penalty_alpha)
        score = prob / penalty
        return score
