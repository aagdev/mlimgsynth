"""
Copyright 2025, Alejandro A. García <aag@zorzal.net>
SPDX-License-Identifier: MIT

Converts CLIP vocabulary merges in a list of token number pairs.
ref: https://github.com/openai/CLIP : clip/simple_tokenizer.py
"""
import gzip

bpe_path = "bpe_simple_vocab_16e6.txt.gz"

# Code copied almost verbatim from CLIP repo
def bytes_to_unicode():
	bs = list(range(ord("!"), ord("~")+1)) \
	   + list(range(ord("¡"), ord("¬")+1)) \
	   + list(range(ord("®"), ord("ÿ")+1))
	cs = bs[:]
	n = 0
	for b in range(2**8):
		if b not in bs:
			bs.append(b)
			cs.append(2**8+n)
			n += 1
	cs = [chr(n) for n in cs]
	return bs, cs

merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
merges = merges[1:49152-256-2+1]
merges = [tuple(merge.split()) for merge in merges]

vocab = list(bytes_to_unicode()[1])
vocab = vocab + [v+'</w>' for v in vocab]
for merge in merges:
	vocab.append(''.join(merge))
vocab.extend(['<|startoftext|>', '<|endoftext|>'])

encoder = dict(zip(vocab, range(len(vocab))))
decoder = {v: k for k, v in encoder.items()}
#bpe_ranks = dict(zip(merges, range(len(merges))))

for left, right in merges:
	l = encoder[left]
	r = encoder[right]
	print("{%d, %d}," % (l, r))
