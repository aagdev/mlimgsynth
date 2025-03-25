"""
Copyright 2024, Alejandro A. García <aag@zorzal.net>
SPDX-License-Identifier: MIT

Minimal module to write PNG images without any external dependencies.
"""
import zlib
import struct

def chunk_write(f, type_str, data):
	ct = type_str.encode("ASCII")
	f.write( struct.pack(">I", len(data)) )
	f.write( ct )
	f.write( data )
	f.write( struct.pack(">I", zlib.crc32(data, zlib.crc32(ct))) )

def ihdr_make(w, h, ch):
	color_type = 6 if ch == 4 else 2 if ch == 3 else 4 if ch == 2 else 0
	out = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
	return out

def data_filter(data, s, h):
	fdata = bytes()
	for y in range(h):
		line = data[s*y:s*(y+1)]
		fdata += b"\0"  # No filter, raw data
		fdata += line
	return fdata

def png_write(f, data, w, h, ch=3, clvl=-1, stride=None, texts=[]):
	# Signature
	f.write(b"\x89PNG\r\n\x1a\n")
	# Header
	ihdr = ihdr_make(w, h, ch)
	chunk_write(f, "IHDR", ihdr)
	# Text chunks
	for name, text in texts:
		text_data = name.encode("utf8") + b"\0" + text.encode("utf8")
		chunk_write(f, "tEXt", text_data)
	# Image data
	fdata = data_filter(data, stride or w*ch, h)
	cdata = zlib.compress(fdata, clvl)
	chunk_write(f, "IDAT", cdata)
	# End
	chunk_write(f, "IEND", bytes())
#end

# Minimal test
if __name__ == "__main__":
	w = 40
	h = 20
	ch = 3
	data = bytes([x*6*(c==0)+y*12*(c==2)
		for y in range(h) for x in range(w) for c in range(ch)])
	with open("minimal_png_test.png", "wb") as f:
		png_write(f, data, w, h, ch, texts=[("source", "minimal_png.py")])
