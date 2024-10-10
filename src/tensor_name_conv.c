/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "tensor_name_conv.h"

static inline
int char_sep_is(char c) {
	return c == '.' || c == '_' || c == '/';
}

// Custom prefix match that matches '.' to '_' and '/' also
int tnconv_prefix_match(const StrSlice ss, const StrSlice pre)
{
	if (!(ss.s >= pre.s)) return 0;
	for (size_t i=0; i<pre.s; ++i) {
		char b = pre.b[i], a = ss.b[i];
		if (!( b == a || (b == '.' && char_sep_is(a)) ))
			return 0;
	}
	return 1;
}

int tnconv_prefix_trim(StrSlice *pss, const StrSlice pre)
{
	if (!tnconv_prefix_match(*pss, pre)) return 0;
	pss->b += pre.s;
	pss->s -= pre.s;
	return 1;
}

int tnconv_prefix_match_replace(StrSlice *pss, DynStr *pres, const StrSlice pre,
	const StrSlice rep)
{
	if (!tnconv_prefix_trim(pss, pre)) return 0;
	if (rep.s) dstr_append(*pres, rep.s, rep.b);
	return 1;
}

int tnconv_number_match(const StrSlice ss)
{
	const char *cur = ss.b, *end = cur + ss.s;
	while (cur < end && '0' <= *cur && *cur <= '9') cur++;
	if (cur == end || !char_sep_is(*cur)) return 0;
	cur++;
	return cur - ss.b;
}

int tnconv_number_match_push(StrSlice *pss, DynStr *pres)
{
	int r = tnconv_number_match(*pss);
	if (r <= 1) return 0;
	dstr_append(*pres, r-1, pss->b);
	dstr_push(*pres, '.');
	pss->b += r;
	pss->s -= r;
	return 1;
}

int tnconv_number_match_get(StrSlice *pss, int *out)
{
	int r = tnconv_number_match(*pss);
	if (r <= 0) return 0;
	if (out) *out = atoi(pss->b);
	pss->b += r;
	pss->s -= r;
	return 1;
}

#define MATCH(S) \
	tnconv_prefix_match(name, strsl_static(S))

#define MATCH_PUSH(S) \
	tnconv_prefix_match_replace(&name, out, strsl_static(S), strsl_static(S))

#define MATCH_REP(S,R) \
	tnconv_prefix_match_replace(&name, out, strsl_static(S), strsl_static(R))

#define MATCH_NUM_PUSH() \
	tnconv_number_match_push(&name, out)

//TODO: find better name: compvis?
int tnconv_clip_1(StrSlice name, DynStr *out)
{
	int R=0;

	if (MATCH_REP("transformer.text_model.", "text.")) {
		if (MATCH_REP("embeddings.", "embed.")) {
			if      (MATCH_REP("position_embedding.", "position.")) R=1;
			else if (MATCH_REP("token_embedding.", "token.")) R=1;
		}
		else if (MATCH_PUSH("encoder.layers.")) {
			MATCH_NUM_PUSH();
			if      (MATCH_REP("layer_norm1.", "norm1.")) R=1;
			else if (MATCH_REP("layer_norm2.", "norm2.")) R=1;
			else if (MATCH_REP("self_attn.", "attn.")) R=1;
			else if (MATCH_PUSH("mlp.")) R=1;
		}
		else if (MATCH_REP("final_layer_norm.", "ln_final.")) R=1;
		else if (MATCH_REP("text_projection", "text_proj")) R=1;
	}
	
	if (R > 0) dstr_append(*out, name.s, name.b);  // Copy tail
	return R;
}

//TODO: find better name: openai?
int tnconv_clip_2(StrSlice name, DynStr *out)
{
	int R=0;

	if (MATCH_REP("model.", "text.")) {
		if      (MATCH_PUSH("ln_final.")) R=1;
		else if (MATCH_REP("token_embedding.", "embed.token.")) R=1;
		else if (MATCH_REP("positional_embedding",
			"embed.position.weight")) R=1;
		else if (MATCH_REP("text_projection", "text_proj")) R=1;
		else if (MATCH_REP("transformer.resblocks.", "encoder.layers.")) {
			MATCH_NUM_PUSH();
			if      (MATCH_REP("ln_1.", "norm1.")) R=1;
			else if (MATCH_REP("ln_2.", "norm2.")) R=1;
			else if (MATCH_PUSH("attn.")) {
				if      (MATCH_PUSH("in_proj_bias")) R=TNCONV_R_QKV_PROJ;
				if      (MATCH_PUSH("in_proj_weight")) R=TNCONV_R_QKV_PROJ;
				else if (MATCH_PUSH("out_proj.")) R=1;
			}
			else if (MATCH_REP("mlp.c_fc.", "mlp.fc1.")) R=1;
			else if (MATCH_REP("mlp.c_proj.", "mlp.fc2.")) R=1;
		}
	}
	
	if (R > 0) dstr_append(*out, name.s, name.b);  // Copy tail
	return R;
}

// Ref.: diffusers/scripts/convert_diffusers_to_original_stable_diffusion.py
// It's unclear to me if this is really diffusers or not...
int tnconv_clip_diffusers(StrSlice name, DynStr *out)
{
	int R=0;

	if (MATCH_REP("text_model.", "text.")) {
		if (MATCH_PUSH("encoder.layers.")) {
			MATCH_NUM_PUSH();
			if      (MATCH_REP("ln_1.", "norm1.")) R=1;
			else if (MATCH_REP("ln_2.", "norm2.")) R=1;
			else if (MATCH_REP("self_attn.", "attn.")) R=1;
			else if (MATCH_PUSH("mlp.")) R=1;
		}
		//else if (MATCH_REP("positional_embedding", "embed.position.weight")) R=1;
		//else if (MATCH_REP("token_embedding.", "embed.token.")) R=1;
		//else if (MATCH_PUSH("ln_final.")) R=1;
		//else if (MATCH_REP("text_projection", "text_proj")) R=1;
	}
	
	if (R > 0) dstr_append(*out, name.s, name.b);  // Copy tail
	return R;
}

int tnconv_vae(StrSlice name, DynStr *out)
{
	int R=0;
		
	if      (MATCH_PUSH("decoder.")) {
		R=1;
		if (MATCH_PUSH("up.") && MATCH_NUM_PUSH() && MATCH_PUSH("block.")
			&& MATCH_NUM_PUSH()) MATCH_REP("nin_shortcut.", "skip_conv.");
	}
	else if (MATCH_PUSH("encoder.")) {
		R=1;
		if (MATCH_PUSH("down.") && MATCH_NUM_PUSH() && MATCH_PUSH("block.")
			&& MATCH_NUM_PUSH()) MATCH_REP("nin_shortcut.", "skip_conv.");
	}
	else if (MATCH_PUSH("quant_conv.")) R=1;
	else if (MATCH_PUSH("post_quant_conv.")) R=1;

	if (R > 0) dstr_append(*out, name.s, name.b);  // Copy tail
	return R;
}

int tnconv_unet_block(StrSlice name, DynStr *out)
{
	int R=0;

	if (MATCH_REP("transformer_blocks.", "transf.")) {
		MATCH_NUM_PUSH();
		if (MATCH_PUSH("attn1.") || MATCH_PUSH("attn2.")) {
			if      (MATCH_REP("to_q.", "q_proj.")) R=1;
			else if (MATCH_REP("to_k.", "k_proj.")) R=1;
			else if (MATCH_REP("to_v.", "v_proj.")) R=1;
			else if (MATCH_REP("to_out.0.", "out_proj.")) R=1;
			R=1;
		}
		else if (MATCH_PUSH("ff.")) {
			if      (MATCH_PUSH("net.0.")) R=1;
			else if (MATCH_PUSH("net.2.")) R=1;
		}
		else if (MATCH_PUSH("norm1.")) R=1;
		else if (MATCH_PUSH("norm2.")) R=1;
		else if (MATCH_PUSH("norm3.")) R=1;
	}
	else if (MATCH_REP("in_layers.0.", "norm1.")) R=1;
	else if (MATCH_REP("in_layers.2.", "conv1.")) R=1;
	else if (MATCH_REP("out_layers.0.", "norm2.")) R=1;
	else if (MATCH_REP("out_layers.3.", "conv2.")) R=1;
	else if (MATCH_REP("emb_layers.1.", "emb_proj.")) R=1;
	else if (MATCH_REP("skip_connection.", "skip_conv.")) R=1;
	else if (MATCH_REP("op.", "conv.")) R=1;
	else if (MATCH_PUSH("norm.")) R=1;
	else if (MATCH_PUSH("proj_in.")) R=1;
	else if (MATCH_PUSH("proj_out.")) R=1;
	else if (MATCH_PUSH("conv.")) R=1;  //upsample
	
	if (R > 0) dstr_append(*out, name.s, name.b);  // Copy tail
	return R;
}

int tnconv_unet(StrSlice name, DynStr *out)
{
	int R=0, n1, n2, n3;

	if      (MATCH_PUSH("time_embed.")) R=1;
	else if (MATCH_REP("label_emb.0.", "label_embed.")) R=1;
	else if (MATCH_REP("input_blocks.0.0.", "in.conv.")) R=1;
	else if (MATCH_REP("out.0.", "out.norm.")) R=1;
	else if (MATCH_REP("out.2.", "out.conv.")) R=1;
	else
	if ((MATCH_REP("input_blocks.", "in.") && MATCH_NUM_PUSH()) ||
		(MATCH_REP("output_blocks.", "out.") && MATCH_NUM_PUSH()) ||
		MATCH_REP("middle_block.", "mid.") )
	{
		MATCH_NUM_PUSH();
		return tnconv_unet_block(name, out);
	}
	// diffusers
	// Ref.: diffusers/scripts/convert_diffusers_to_original_stable_diffusion.py
	else if (MATCH_REP("down_blocks.", "in.")) {
		if (!tnconv_number_match_get(&name, &n1)) return 0;
		if (MATCH_REP("downsamplers.0.conv.", "")) {
			dstr_printfa(*out, "%d.0.op.", 3*(n1+1));
		} else {
			if      (MATCH_REP("attentions.", "")) n2=1;
			else if (MATCH_REP("resnets.", "")) n2=0;
			else return 0;
			if (!tnconv_number_match_get(&name, &n3)) return 0;
			dstr_printfa(*out, "%d.%d.", 3*n1+n3+1, n2);
		}
		return tnconv_unet_block(name, out);
	}
	else if (MATCH_REP("up_blocks.", "out.")) {
		if (!tnconv_number_match_get(&name, &n1)) return 0;
		if (MATCH_REP("upsamplers.0.", "")) {
			dstr_printfa(*out, "%d.%d.", 3*n1+2, n1==0 ? 1 : 2);
		} else {
			if      (MATCH_REP("attentions.", "")) n2=1;
			else if (MATCH_REP("resnets.", "")) n2=0;
			else return 0;
			if (!tnconv_number_match_get(&name, &n3)) return 0;
			dstr_printfa(*out, "%d.%d.", 3*n1+n3, n2);
		}
		return tnconv_unet_block(name, out);
	}
	else if (MATCH_REP("mid_block.", "mid.")) {
		if      (MATCH_REP("attentions.0.", "1.")) {
			return tnconv_unet_block(name, out);
		}
		else if (MATCH_REP("resnets.0.", "0.")) R=1;
		else if (MATCH_REP("resnets.1.", "2.")) R=1;
	}

	if (R > 0) dstr_append(*out, name.s, name.b);  // Copy tail
	return R;
}

int tnconv_sd(StrSlice name, DynStr *out)
{
	int R=0;

	// Text encoding (clip)
	if (MATCH_REP("cond_stage_model.", "clip.")) {
		//sd1
		if (MATCH("transformer.text_model.")) {
			return tnconv_clip_1(name, out);
		}
		//sd2
		else if (MATCH("model.")) {
			return tnconv_clip_2(name, out);
		}
	}
	else if (MATCH_REP("te.", "clip.")) {
		return tnconv_clip_diffusers(name, out);
	}
	//sdxl
	else if (MATCH_REP("conditioner.embedders.0.", "clip.")) {
		return tnconv_clip_1(name, out);
	}
	else if (MATCH_REP("conditioner.embedders.1.", "clip2.")) {
		return tnconv_clip_2(name, out);
	}
	else if (MATCH_REP("te1.", "clip.")) {
		return tnconv_clip_diffusers(name, out);
	}
	else if (MATCH_REP("te2.", "clip2.")) {
		return tnconv_clip_diffusers(name, out);
	}

	// Image-latent codec (vae)
	else if (MATCH_REP("first_stage_model.", "vae.")) {
		return tnconv_vae(name, out);
	}

	// Denoiser (unet)
	else if (MATCH_REP("model.diffusion_model.", "unet.") || 
		MATCH_PUSH("unet.") )
	{
		return tnconv_unet(name, out);
	}

	if (R > 0) dstr_append(*out, name.s, name.b);  // Copy tail
	return R;
}
