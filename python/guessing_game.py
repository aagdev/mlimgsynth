"""
Copyright 2024, Alejandro A. García <aag@zorzal.net>
SPDX-License-Identifier: MIT

Example program using the MLImgSynth library.
Web-based game where you see an AI generated image and have to guess the prompt.
No external modules needed.
"""
import random
import logging
import argparse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qsl

from mlimgsynth import MLImgSynth
from minimal_png import png_write

ADJECTIVES = [
	"red", "blue", "green", "yellow",
]

NOUNS = [
	"lion", "rabbit", "cow", "chicken",
	"cup", "table", "lamp", "book", "car",
]

PLACES = [
	"in the mountains", "on a lake", "in a river", "on a beach", "in a forest",
	"in a city street", "in a cavern"
]

class GuessingGame:
	def __init__(self, mlis):
		self.mlis = mlis
		self.prompt = None
		self.img = None
		self.feat = None
		self.last_score = 0.0
		self.last_guess = ""
		
		self.elements = [ADJECTIVES, NOUNS, PLACES]
		self.prompt_prefix = None
		self.nprompt = None
	#end

	def generate(self):
		self.img = None
		self.last_score = 0.0
		self.last_guess = ""

		self.prompt_elems = [random.choice(elist) for elist in self.elements]
		self.prompt = " ".join(self.prompt_elems)
		logging.debug("Prompt: " + self.prompt)
		#embd, self.feat = self.mlis.clip_text_encode(self.prompt, features=True)

		p = self.prompt
		if self.prompt_prefix:
			p = self.prompt_prefix + " " + p
		self.mlis.option_set("prompt", p)
		if self.nprompt:
			self.mlis.option_set("nprompt", self.nprompt)

		logging.info("Generating image...")
		self.mlis.generate()
		self.img = self.mlis.image_get()
	#end

	def guess_check(self, guess):
		#embd, feat = self.mlis.clip_text_encode(guess, features=True)
		#s = self.feat.similarity(feat)
		elems = guess.split(maxsplit=2)
		elems = [x.strip().lower() for x in elems]
		score = sum(int(x == y) for x, y in zip(elems, self.prompt_elems))
		score /= len(self.prompt_elems)
		self.last_guess = guess
		self.last_score = score
		return score
	#end

	def image_png_write(self, f):
		png_write(f, self.img.data, self.img.w, self.img.h, self.img.c)
	#end
#end

PAGE = b"""
<html>
<head>
	<title>Guessing Game</title>
	<style>
html, body {
	max-width: max-content;
	margin: 0 auto;
}
	</style>
</head>
<body>
	<h1>Guessing Game</h1>
	<form style="display: inline;">
		Try to guess the image prompt:<br/>
		<input type="text" name="guess" size=40 placeholder="red car on a beach" value="{{last_guess}}"/>
		<input type="submit" value="Guess">
	</form>
	Score: {{last_score}}
	<form style="display: inline;">
		<input type="hidden" name="new" value="1"/>
		<input type="submit" value="New Image">
	</form>
	<br/>
	<img src="/image.png" alt="Image to guess"/>
</body>
</html>
"""

class GuessingGameWebHandler(BaseHTTPRequestHandler):
	def page_main(self):
		self.send_response(200)
		self.send_header('Content-type', 'text/html')
		self.end_headers()
		last_score = format(self.server.game.last_score, ".2f").encode("ascii")
		last_guess = self.server.game.last_guess.encode("ascii")
		page = PAGE.replace(b"{{last_score}}", last_score) \
		           .replace(b"{{last_guess}}", last_guess)
		self.wfile.write(page)

	def page_image(self):
		self.send_response(200)
		self.send_header('Content-type', 'image/png')
		self.end_headers()
		self.server.game.image_png_write(self.wfile)

	def page_not_found(self):
		self.send_response(404)
		self.send_header('Content-type', 'text/plain')
		self.end_headers()
		self.wfile.write(b"404 Not Found\n")

	def do_GET(self):
		url = urlparse(self.path)
		if url.path == "/":
			kv = parse_qsl(url.query)
			if kv:
				if kv[0][0] == "new":
					self.server.game.generate()
				elif kv[0][0] == "guess":
					self.server.game.guess_check(kv[0][1])
			self.page_main()
		elif url.path == "/image.png":
			self.page_image()
		else:
			self.page_not_found()
#end

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-H", "--host", default="127.0.0.1")
	parser.add_argument("-P", "--port", type=int, default=8000)
	parser.add_argument("-m", "--model", required=True,
		help="Image generation model file path.")
	parser.add_argument("-p", "--prompt-prefix")
	parser.add_argument("-n", "--negative-prompt")
	parser.add_argument("-g", "--genopt",
		help="List of image generation options like: steps=12:method=euler:...")
	parser.add_argument("--no-browser", action="store_true",
		help="Do not open the page in a browser.")
	parser.add_argument("-D", "--debug", action="store_true")
	args = parser.parse_args()

	
	logging.basicConfig(
		level=logging.DEBUG if args.debug else logging.INFO,
		format="[GAME] %(levelname)s %(message)s" )
	
	mlis = MLImgSynth()
	mlis.option_set("log-level", "debug" if args.debug else "info")
	mlis.option_set("model", args.model)

	if args.genopt:
		for kv in args.genopt.split(":"):
			k,_,v = kv.partition("=")
			mlis.option_set(k, v)
	
	game = GuessingGame(mlis)
	game.prompt_prefix = args.prompt_prefix
	game.nprompt = args.negative_prompt
	game.generate()

	httpd = HTTPServer((args.host, args.port), GuessingGameWebHandler)
	httpd.game = game
	logging.info("Listening on %s:%s", args.host, args.port)
	if not args.no_browser and args.host == "127.0.0.1":
		httpd.server_activate()
		webbrowser.open("http://127.0.0.1:%d" % args.port)
	httpd.serve_forever()
#end

if __name__ == '__main__':
	main()
