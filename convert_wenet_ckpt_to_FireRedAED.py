import torch

import copy
import argparse
    

def convert_to_fireredaed_state_dict(wenet_state_dict):
    fireredaed_state_dict = {}
    unused = []
    print("===================== start CKPT Conversion =========================")
    
    for name in wenet_state_dict.keys():
        original_name = copy.deepcopy(name)

        name = name.replace("embed.conv", "input_preprocessor.conv")
        name = name.replace("embed.out.0", "input_preprocessor.out")
        
        if original_name == "decoder.embed.0.weight":
            name = "decoder.tgt_word_emb.weight"
        if original_name == "decoder.output_layer.weight":
            name = "decoder.tgt_word_prj.weight"
        if original_name == "encoder.embed.pos_enc.pe":
            name = "encoder.positional_encoding.pe"
        if original_name == "decoder.embed.1.pe":
            name = "decoder.positional_encoding.pe"
        
        name = name.replace("encoder.encoders", "encoder.layer_stack")
        name = name.replace("decoder.decoders", "decoder.layer_stack")
        name = name.replace("decoder.after_norm", "decoder.layer_norm_out")
        
        # encoder attn
        if original_name.startswith("encoder"):
            name = name.replace(".self_attn.linear_q", ".mhsa.w_qs")
            name = name.replace(".self_attn.linear_k", ".mhsa.w_ks")
            name = name.replace(".self_attn.linear_v", ".mhsa.w_vs")
            name = name.replace(".self_attn.linear_out", ".mhsa.fc")
            name = name.replace(".self_attn.pos_bias_u", ".mhsa.pos_bias_u")
            name = name.replace(".self_attn.pos_bias_v", ".mhsa.pos_bias_v")
            name = name.replace(".self_attn.linear_pos", ".mhsa.linear_pos")
            
        # decoder attn
        if original_name.startswith("decoder"):
            name = name.replace(".src_attn.linear_q", ".cross_attn.w_qs")
            name = name.replace(".src_attn.linear_k", ".cross_attn.w_ks")
            name = name.replace(".src_attn.linear_v", ".cross_attn.w_vs")
            name = name.replace(".src_attn.linear_out", ".cross_attn.fc")
            name = name.replace(".self_attn.linear_q", ".self_attn.w_qs")
            name = name.replace(".self_attn.linear_k", ".self_attn.w_ks")
            name = name.replace(".self_attn.linear_v", ".self_attn.w_vs")
            name = name.replace(".self_attn.linear_out", ".self_attn.fc")
            
        # decoder mlp
        if original_name.startswith("decoder"):
            name = name.replace(".feed_forward.", ".mlp.")
        
        # encodr mlp
        if original_name.startswith("encoder"):
            name = name.replace(".feed_forward_macaron.w_1", ".ffn1.net.1")
            name = name.replace(".feed_forward_macaron.w_2", ".ffn1.net.4")
            name = name.replace(".feed_forward.w_1", ".ffn2.net.1")
            name = name.replace(".feed_forward.w_2", ".ffn2.net.4")

        # decoder pre norm
        if original_name.startswith("decoder"):
            name = name.replace(".norm1.", ".self_attn_norm.")
            name = name.replace(".norm2.", ".cross_attn_norm.")
            name = name.replace(".norm3.", ".mlp_norm.")
            
        # encoder pre norm
        if original_name.startswith("encoder"):
            name = name.replace(".norm_ff_macaron.", ".ffn1.net.0.")
            name = name.replace(".self_attn.layer_norm_q.", ".mhsa.layer_norm_q.")
            name = name.replace(".self_attn.layer_norm_k.", ".mhsa.layer_norm_k.")
            name = name.replace(".self_attn.layer_norm_v.", ".mhsa.layer_norm_v.")
            name = name.replace(".norm_conv.", ".conv.pre_layer_norm.")
            name = name.replace(".norm_ff", ".ffn2.net.0")
            name = name.replace(".norm_final.", ".layer_norm.")
            name = name.replace(".norm_final.", ".layer_norm.")
        
        # encoder conv
        if original_name.startswith("encoder"):
            name = name.replace(".conv_module.", ".conv.")
            name = name.replace(".norm.", ".batch_norm.")
            
        if "decoder" in name:
            name = name.replace("norm2", "cross_attn_ln", )
            name = name.replace("norm3", "mlp_ln")
        else:
            name = name.replace("norm2", "mlp_ln")
            
        print(f"name {original_name} ==> {name}")
        print("type  {} ==> torch.float32".format(
            wenet_state_dict[original_name].dtype))
        print("shape {}\n".format(wenet_state_dict[original_name].shape))
        
        if original_name == name:
            unused.append(name)
        else:
            fireredaed_state_dict[name] = wenet_state_dict[original_name].float()
            
    for name in unused:
        print(f"NOTE!!! drop {name}")
            
    print(
        "DONE\n===================== End CKPT Conversion =========================\n"
    )
    
    return fireredaed_state_dict


def generate_model_args():
    args = argparse.Namespace()
    
    args.blank = '<blank>'
    args.pad = '<pad>'
    args.sos = '<sos>'
    args.eos = '<eos>'
    args.input_length_max = 60.0
    args.input_length_min = 0.1
    args.output_length_max = 250
    args.output_length_min = 1
    args.idim = 80
    args.odim = 7832
    args.n_layers_enc = 16
    args.n_head = 20
    args.d_model = 1280
    args.d_inner = 5120
    args.residual_dropout = 0.1
    args.pe_maxlen = 5000
    args.dropout_rate = 0.1
    args.kernel_size = 33
    args.n_layers_dec = 16
    args.unk = '<unk>' 
    args.blank_id = 0 
    args.sos_id = 3 
    args.eos_id = 4 
    args.pad_id = 2
    
    return args
    

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adapting the wenet-fine-tuned fireredaed model to FireRedASR"
    )
    
    parser.add_argument(
        "--wenet_model",
        type=str,
        required=True,
        help=""
    )
    parser.add_argument(
        "--fireredaed_model",
        type=str,
        required=True,
        help=""
    )

    return parser.parse_args()


def main():
    args = parse_args()
    
    wenet_finetune_fireredaed_state_dict = torch.load(args.wenet_model)
    model_state_dict = convert_to_fireredaed_state_dict(wenet_finetune_fireredaed_state_dict)
    model_args = generate_model_args()
    
    torch.save(
        {
            "args": model_args,
            "model_state_dict": model_state_dict
        },
        args.fireredaed_model
    )
    

if __name__ == "__main__":
    main()