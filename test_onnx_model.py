from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer

import onnxruntime as ort
import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Tuple, List, Dict
import argparse
import os
import time
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger_stream_hander = logging.StreamHandler()
logger_stream_hander.setLevel("INFO")
logger.addHandler(logger_stream_hander)


INF = 1e10


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()
    
    
def set_finished_beam_score_to_zero(scores, is_finished):
    NB, B = scores.size()
    is_finished = is_finished.float()
    mask_score = torch.tensor([0.0] + [-INF]*(B-1)).float()
    mask_score = mask_score.view(1, B).repeat(NB, 1)
    return scores * (1 - is_finished) + mask_score * is_finished


def set_finished_beam_y_to_eos(ys, is_finished, eos_id):
    is_finished = is_finished.long()
    return ys * (1 - is_finished) + eos_id * is_finished


class FireRedASROnnxModel:
    def __init__(
        self, 
        encoder_path: str, 
        decoder_path: str,
        cmvn_file: str,
        dict_file: str, 
        spm_model_path: str,
        providers=["CPUExecutionProvider"]
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1
        # session_opts.log_severity_level = 1
        self.session_opts = session_opts
        
        # NOTE: 参考whisper设置的最大的解码长度
        # FireRedASR-AED 模型支持的最长语音为 60s
        # ref: https://github.com/FireRedTeam/FireRedASR?tab=readme-ov-file#input-length-limitations
        self.decode_max_len = 448
        
        self.decoder_hidden_dim = 1280
        self.num_decoder_blocks = 16
        self.blank_id = 0
        self.sos_id = 3
        self.eos_id = 4
        self.pad_id = 2
        
        self.feature_extractor = ASRFeatExtractor(cmvn_file)
        self.tokenizer = ChineseCharEnglishSpmTokenizer(dict_file, spm_model_path)
        self.encoder = None
        self.decoder = None
        
        self.init_encoder(encoder_path, providers)
        self.init_decoder(decoder_path, providers)
        
    def init_encoder(self, encoder_path, providers=None):
        start_time = time.time()
        self.encoder = ort.InferenceSession(
            encoder_path,
            sess_options=self.session_opts,
            providers=providers
        )
        end_time = time.time()
        logger.info(f"load encoder cost {end_time - start_time} seconds")
    
    def init_decoder(self, decoder_path, providers=None):
        start_time = time.time()
        self.decoder = ort.InferenceSession(
            decoder_path,
            sess_options=self.session_opts,
            providers=providers
        )
        end_time = time.time()
        logger.info(f"load decoder cost {end_time - start_time} seconds")
    
    def run_encoder(self, input: np.ndarray, 
                    input_length: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor]:
        n_layer_cross_k, n_layer_cross_v, cross_attn_mask = self.encoder.run(
            None,
            {
                self.encoder.get_inputs()[0].name: input,
                self.encoder.get_inputs()[1].name: input_length
            }
        )
        return (
            n_layer_cross_k,
            n_layer_cross_v,
            cross_attn_mask
        )
        
    def decode_one_token(
        self,
        tokens: np.ndarray,
        n_layer_self_k_cache: np.ndarray,
        n_layer_self_v_cache: np.ndarray,
        n_layer_cross_k_cache: np.ndarray,
        n_layer_cross_v_cache: np.ndarray,
        offset: np.ndarray,
        self_attn_mask: np.ndarray,
        cross_attn_mask: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor]:
        logits, out_n_layer_self_k_cache, out_n_layer_self_v_cache = self.decoder.run(
            None,
            {
                self.decoder.get_inputs()[0].name: tokens,
                self.decoder.get_inputs()[1].name: n_layer_self_k_cache,
                self.decoder.get_inputs()[2].name: n_layer_self_v_cache,
                self.decoder.get_inputs()[3].name: n_layer_cross_k_cache,
                self.decoder.get_inputs()[4].name: n_layer_cross_v_cache,
                self.decoder.get_inputs()[5].name: offset,
                self.decoder.get_inputs()[6].name: self_attn_mask,
                self.decoder.get_inputs()[7].name: cross_attn_mask,
            }
        )
        return (
            logits,
            out_n_layer_self_k_cache,
            out_n_layer_self_v_cache
        )
    
    def run_decoder(
        self,
        n_layer_cross_k, 
        n_layer_cross_v, 
        cross_attn_mask,
        beam_size,
        nbest
    ):
        
        num_layer, batch_size, Ti, encoder_out_dim = n_layer_cross_k.shape
        encoder_out_length = cross_attn_mask.shape[-1]
        
        cross_attn_mask = torch.from_numpy(cross_attn_mask).to(torch.float32)
        cross_attn_mask = cross_attn_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1
        ).view(beam_size * batch_size, -1, encoder_out_length)
        
        n_layer_cross_k = torch.from_numpy(n_layer_cross_k)
        n_layer_cross_v = torch.from_numpy(n_layer_cross_v)
        n_layer_cross_k = n_layer_cross_k.unsqueeze(2).repeat(
            1, 1, beam_size, 1, 1
        ).view(num_layer, beam_size * batch_size, Ti, encoder_out_dim)
        n_layer_cross_v = n_layer_cross_v.unsqueeze(2).repeat(
            1, 1, beam_size, 1, 1
        ).view(num_layer, beam_size * batch_size, Ti, encoder_out_dim)

        prediction_tokens = torch.ones(
            beam_size * batch_size, 1).fill_(self.sos_id).long()
        tokens = prediction_tokens
        offset = torch.zeros(1, dtype=torch.int64)
        n_layer_self_k_cache, n_layer_self_v_cache = self.get_initialized_self_cache(
            batch_size, beam_size
        )
        
        scores = torch.tensor([0.0] + [-INF]*(beam_size - 1)).float()
        scores = scores.repeat(batch_size).view(batch_size * beam_size, 1)
        is_finished = torch.zeros_like(scores)
        
        results = [self.sos_id]
        for i in range(self.decode_max_len):
            self_attn_mask = torch.empty(
                batch_size * beam_size, 
                prediction_tokens.shape[-1], prediction_tokens.shape[-1]
            ).fill_(-np.inf).triu_(1)
            self_attn_mask = self_attn_mask[:, -1:, :]

            logits, n_layer_self_k_cache, n_layer_self_v_cache = self.decode_one_token(
                to_numpy(tokens),
                to_numpy(n_layer_self_k_cache),
                to_numpy(n_layer_self_v_cache),
                to_numpy(n_layer_cross_k),
                to_numpy(n_layer_cross_v),
                to_numpy(offset),
                to_numpy(self_attn_mask),
                to_numpy(cross_attn_mask)
            )
            
            offset += 1
            logits = torch.from_numpy(logits)
            
            logits = logits.squeeze(1)
            t_scores = F.log_softmax(logits, dim=-1)
            t_topB_scores, t_topB_ys = torch.topk(t_scores, k=beam_size, dim=1)
            t_topB_scores = set_finished_beam_score_to_zero(t_topB_scores, is_finished)
            t_topB_ys = set_finished_beam_y_to_eos(t_topB_ys, is_finished, self.eos_id)
            
            scores = scores + t_topB_scores
            
            scores = scores.view(batch_size, beam_size * beam_size)
            scores, topB_score_ids = torch.topk(scores, k=beam_size, dim=1)
            scores = scores.view(-1, 1)
            
            topB_row_number_in_each_B_rows_of_ys = torch.div(
                topB_score_ids, beam_size).view(batch_size * beam_size)
            stride = beam_size * torch.arange(batch_size).view(
                batch_size, 1).repeat(1, beam_size).view(batch_size * beam_size)
            topB_row_number_in_ys = topB_row_number_in_each_B_rows_of_ys.long() + stride.long()
            
            prediction_tokens = prediction_tokens[topB_row_number_in_ys]
            t_ys = torch.gather(
                t_topB_ys.view(batch_size, beam_size * beam_size), 
                dim=1, index=topB_score_ids
            ).view(beam_size * batch_size, 1)
            
            tokens = t_ys
        
            prediction_tokens = torch.cat((prediction_tokens, t_ys), dim=1)
            
            n_layer_self_k_cache = torch.from_numpy(n_layer_self_k_cache)
            n_layer_self_v_cache = torch.from_numpy(n_layer_self_v_cache)
            
            for i, self_k_cache in enumerate(n_layer_self_k_cache):
                n_layer_self_k_cache[i] = n_layer_self_k_cache[i][topB_row_number_in_ys]
            
            for i, self_v_cache in enumerate(n_layer_self_v_cache):
                n_layer_self_v_cache[i] = n_layer_self_v_cache[i][topB_row_number_in_ys]
                
            is_finished = t_ys.eq(self.eos_id)
            if is_finished.sum().item() == beam_size * batch_size:
                break
            
        scores = scores.view(batch_size, beam_size)
        prediction_valid_token_lengths = torch.sum(
            torch.ne(
                prediction_tokens.view(batch_size, beam_size, -1), 
                self.eos_id),
            dim=-1
        ).int()
       
        nbest_scores, nbest_ids = torch.topk(scores, k=nbest, dim=1)
        index = nbest_ids + beam_size * torch.arange(batch_size).view(batch_size, 1).long()
        nbest_prediction_tokens = prediction_tokens.view(batch_size * beam_size, -1)[index.view(-1)]
        nbest_prediction_tokens = nbest_prediction_tokens.view(batch_size, nbest_ids.size(1), -1)
        nbest_prediction_valid_token_lengths = prediction_valid_token_lengths.view(
            batch_size * beam_size)[index.view(-1)].view(batch_size, -1)
        nbest_hyps: List[List[Dict[str, torch.Tensor]]] = []
        for i in range(batch_size):
            i_best_hyps: List[Dict[str, torch.Tensor]] = []
            for j, score in enumerate(nbest_scores[i]):
                hyp = {
                    "token_ids": nbest_prediction_tokens[i, j, 1:nbest_prediction_valid_token_lengths[i, j]],
                    "score": score
                }
                i_best_hyps.append(hyp)
            nbest_hyps.append(i_best_hyps)

        return nbest_hyps
        
    def get_initialized_self_cache(self, 
                                   batch_size, 
                                   beam_size
                                   ) -> Tuple[Tensor, Tensor]:
        n_layer_self_k_cache = torch.zeros(
            self.num_decoder_blocks,
            batch_size * beam_size,
            self.decode_max_len,
            self.decoder_hidden_dim,
        )
        n_layer_self_v_cache = torch.zeros(
            self.num_decoder_blocks,
            batch_size * beam_size,
            self.decode_max_len,
            self.decoder_hidden_dim,
        )
        return n_layer_self_k_cache, n_layer_self_v_cache
 
    def transcribe(self, 
                   batch_wav_path: List[str],
                   beam_size: int = 1,
                   nbest: int = 1
                ) -> List[Dict]:
        feats, lengths, wav_durations = self.feature_extractor(batch_wav_path)
        start_time = time.time()
        n_layer_cross_k, n_layer_cross_v, cross_attn_mask = self.run_encoder(
            to_numpy(feats),
            to_numpy(lengths)
        )
        nbest_hyps = self.run_decoder(n_layer_cross_k,
                                      n_layer_cross_v,
                                      cross_attn_mask,
                                      beam_size,
                                      nbest)
        transcribe_durations = time.time() - start_time
        results: List[Dict] = []
        for wav, hyp in zip(batch_wav_path, nbest_hyps):
            hyp = hyp[0]
            hyp_ids = [int(id) for id in hyp["token_ids"].cpu()]
            score = hyp["score"].item()
            text = self.tokenizer.detokenize(hyp_ids)
            results.append(
                {
                    "wav": wav,
                    "text": text,
                    "score": score
                }
            )
            
        return results, wav_durations, transcribe_durations   
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="FireRedASROnnxModel Test")
    parser.add_argument(
        "--encoder", 
        type=str, 
        required=True,
        help="Path to onnx encoder"
    )
    parser.add_argument(
        "--decoder", 
        type=str, 
        required=True,
        help="Path to onnx decoder"
    )
    parser.add_argument(
        "--provider",
        choices=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    parser.add_argument(
        "--cmvn",
        type=str,
        required=True,
        help="Path to cmvn"
    )
    parser.add_argument(
        "--dict",
        type=str,
        required=True,
        help="Path to dict"
    )
    parser.add_argument(
        "--spm_model",
        type=str,
        required=True,
        help="Path to spm model"
    )
    parser.add_argument(
        "--wavlist",
        type=str,
        required=True,
        help="File to wav path list"
    )
    parser.add_argument(
        "--hypo",
        type=str,
        required=True,
        help="File of hypos"
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=3,
        help=""
    )
    parser.add_argument(
        "--nbest",
        type=int,
        default=1,
        help=""
    )
    
    return parser.parse_args()
    
    
def parse_wavlist(wavlist: str):
    wavpaths = []
    with open(wavlist) as f:
        for line in f:
            line = line.strip()
            if not os.path.exists(line):
                print(f"{line} doesn't exist.")
                continue
            wavpaths.append(line)
            
    return wavpaths
    

def main():
    args = parse_args()
    print(args)
    
    onnx_model = FireRedASROnnxModel(args.encoder,
                                     args.decoder,
                                     args.cmvn,
                                     args.dict,
                                     args.spm_model,
                                     [args.provider])
    
    wf = open(args.hypo, "wt")
    wavlist = parse_wavlist(args.wavlist)

    total_wav_durations = 0
    total_transcribe_durations = 0
    for wav in wavlist:
        batch_wav = [wav]
        results, wav_durations, transcribe_durations = onnx_model.transcribe(
            batch_wav, args.beam_size, args.nbest)
        
        wav_durations = sum(wav_durations)
        total_wav_durations += wav_durations
        total_transcribe_durations += transcribe_durations
        logger.info(f"{batch_wav}")
        logger.info(f"Durations: {wav_durations}")
        logger.info(f"Transcribe Durations: {transcribe_durations}")
        rtf = transcribe_durations / wav_durations
        logger.info(f"(Real time factor) RTF: {rtf}")
        for result in results:
            logger.info(f"wav: {result['wav']}")
            logger.info(f"text: {result['text']}")
            logger.info(f"score: {result['score']}")
            logger.info("")
            wf.write(f"{result['text']} ({result['wav']})\n")
            
    logger.info(f"total wav durations: {total_wav_durations}")
    logger.info(f"total transcribe durations: {total_transcribe_durations}")
    avg_ref = total_transcribe_durations / total_wav_durations
    logger.info(f"AVG RTF: {avg_ref}")
    
    wf.close()
    

if __name__ == "__main__":
    main()