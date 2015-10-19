# Theresa Breiner CIS 530 HW3
from collections import *
import operator
from os import listdir

def parse_taggedfile(wsjfile, tagmap):
	"""Parses the raw PTB-tagged wsjfile (string path), which has token/pos pairs 
	separated by spaces or newlines, and sentences separated by one or
	more blank lines. Ignores brackets or ===== lines in the raw file.
	Converts all tokens to lowercase first. tagmap is a dict that maps PTB tags
	to Google Universal tags.
	Returns a list of lists of (token, Google pos) tuples, where each list is the tuples from
	a sentence."""
	return get_pairs(wsjfile, tagmap, True, False)

def get_pairs(filename, tagmap, convert, doCount):
	"""Helper method to parse a wsj file and return a list of list of (token, pos) tuples.
	If convert is set to True the pos tags will be converted to the Google ones.
	If doCount is set to True will count pos tags along the way and return a dict of dicts of the tokens
	and their pos tags."""
	f = open(filename, "r")
	lines = f.readlines()
	f.close()
	whole = []
	sentence = []
	counts = defaultdict(dict)
	append_sent = True	# true if we have already appended the sent
	for line in lines:
		line = line.strip()
		# check if a sentence separator
		if "=====" in line or len(line) == 0:
			# if we haven't appended sentence already, do so
			if not append_sent:
				whole.append(sentence[:])
				sentence = []
				append_sent = True
		else:
			# contains content
			append_sent = False	# we have a chunk from an unappended sentence
			pairs = line.split()
			# print pairs
			for chunk in pairs:
				ind = chunk.rfind("/")
				# print ind
				if ind != -1:
					token = chunk[:ind].lower()
					tag = chunk[ind+1:]
					# append (lowercased token, pos tag)
					if convert:
						# use Google pos tag
						sentence.append((token, tagmap[tag]))
					else:
						# use PTB pos tag
						sentence.append((token, tag))
						# if we need to count pos tags per token
						if doCount:
							try:
								counts[token][tag] += 1
							except KeyError:
								counts[token][tag] = 1

	if doCount:
		return counts
	# if we weren't doing counts, just return list of list of pairs
	return whole

####################### Part 2 ############################

def create_mft_dict(filelist):
	"""Creates a dictionary of the most frequent POS tag for each unique lowercase
	token that appears in the unparsed input files. filelist is a list of str file names.
	Returns a dict containing str token keys and str pos tag values."""

	# create an empty dict of token keys to dict values where tags are mapped to int counts
	counts = defaultdict(dict)
	for f in filelist:
		# get the counts per tag per token for each file (not converting to Google tags)
		one_file_counts = get_pairs(f, None, False, True)
		for (token, tags) in one_file_counts.items():
			for (tag, c) in tags.items():
				# update the counts of that tag for that token
				try:
					counts[token][tag] += c
				except KeyError:
					counts[token][tag] = c
	# create mft_dict
	mft_dict = defaultdict(str)
	for token in counts:
		# sort the possible tags by frequency
		sortedTags = sorted(counts[token].items(), key=operator.itemgetter(1))
		sortedTags.reverse()
		# save the highest frequency tag as the tag for the token
		mft_dict[token] = sortedTags[0][0]
	return mft_dict

def run_mft_baseline(testfilelist, mftdict, poslookup):
	"""Takes the mftdict mapping tokens to their most frequent tag in training, and
	the poslookup dict mapping PTB tags to Google Universal tags, and predicts
	the Google Universal POS tag of each token in the list of unparsed input filepaths.
	After making predictions, returns the float accuracy of the predicted tags (the number
		of correctly predicted tags in the test files divided by the total number of predicted tags)"""
	
	# add an empty key/value pair to poslookup for words that weren't in mftdict
	poslookup[""] = ""
	# track totals for calculating accuracy
	total_words = 0
	correctly_predicted = 0
	for f in testfilelist:
		# gather the list of sentences split into lists of (token, tag) tuples
		sents = parse_taggedfile(f, poslookup) # uses Google tags
		for sent in sents:
			for pair in sent:
				# print pair
				# track number of words examined
				total_words += 1
				# print poslookup[mftdict[pair[0]]]
				if pair[1] == poslookup[mftdict[pair[0]]]:
					# print "yes", correctly_predicted + 1
					# if the actual tag is what the mftdict stores for that token, it's correct
					correctly_predicted += 1
				# else:
					# print "no"
	return correctly_predicted * 1.0 / total_words

####################### Part 3 #############################

def prep_data(dirname, outfile, windowsize, tagmap, vocab):
	"""Reads data from all raw POS-tagged files in directory dirname and converts to an intermediate
	format in outfile. The format of outfile consists of one <tag> <context window> pair per line,
	tab separated, with the words in the context window separated by a space.
	Replaces words that are not within vocab with <UNK> and pads sentence boundaries with <s> tokens.
	The context window never spans more than one sentence. Windowsize is an odd positive integer, tagmap is
	a dict mapping PTB tags to Google Universal tags, vocab is a set containing str tokens."""

	# get all .pos files that are in dirname
	filepaths = [dirname + "/" + f for f in listdir(dirname) if ".pos" in f]
	out = open(outfile, "w")
	# loop through each file
	for fp in filepaths:
		sents = parse_taggedfile(fp, tagmap)
		for sent in sents:
			sent = replace_unknown(sent, vocab)
			for ind in range(len(sent)):
				out.write(get_window(sent, ind, windowsize))
	out.close()

def replace_unknown(sent, vocab):
	"""Helper method to replace all tokens in the given sent (a list of (token, pos) tuples)
	with "<UNK>" if they are not in the vocab. Returns the edited sent."""

	fixed = []
	for pair in sent:
		if pair[0] not in vocab:
			fixed.append(("<UNK>", pair[1]))
		else:
			fixed.append(pair)
	return fixed


def get_window(sent, ind, windowsize):
	"""Helper method to return a string to be written to the output file in prep_data.
	sent is a list of (token, pos) tuples, ind is the index of the current target, and
	windowsize is the size of the window (an odd positive integer).
	Returns a string such as "NOUN\ta cat saw\n" for window of 3."""

	# first put POS
	line = sent[ind][1] + "\t"
	for i in range(ind - windowsize / 2, ind + windowsize / 2 + 1):
		# if out of bounds, write the sentence delimiter
		if i < 0 or i >= len(sent):
			line += "<s> "
		else:
			line += sent[i][0] + " "
	line += "\n"
	return line

def get_vocab(dirname, min_occurrence):
	"""Helper method to find the vocabulary, which will be all tokens in the files
	within the dirname that occur at least min_occurrence times. Returns the vocabulary
	as a set of string tokens."""

	# get all filepaths in dirname that are .pos files
	filepaths = [dirname + "/" + f for f in listdir(dirname) if ".pos" in f]
	counts = defaultdict(int)
	vocab = set()
	for fp in filepaths:
		f = open(fp, "r")
		lines = f.readlines()
		for line in lines:
			line = line.strip()
			# only consider lines that have token/pos
			if len(line) != 0 and "=====" not in line:
				pairs = line.split()
				for chunk in pairs:
					# find the token/pos divider
					ind = chunk.rfind("/")
					# if there is a divider, ie it is a token/pos pair
					if ind != -1:
						token = chunk[:ind]
						counts[token] += 1
						# once we have at least min_occurrence, add to vocab
						if counts[token] == min_occurrence:
							vocab.add(token)
		f.close()
	return vocab





if __name__ == '__main__':
	mapf = open("en-ptb.map", "r")
	tagmap = { line.split()[0]:line.split()[1] for line in mapf.readlines()}
	# print tagmap.items()[0]
	# print parse_taggedfile("train/wsj_0100.pos", tagmap)
	# filelist = ["train/wsj_0" + str(n) + ".pos" for n in range(100, 500)]
	# print filelist
	# mft = create_mft_dict(filelist)
	# tests = ["test/wsj_" + str(n) + ".pos" for n in range(2200, 2500)]
	# print tests
	# print run_mft_baseline(tests, mft, tagmap)
	
	vocab = get_vocab("train", 8)
	print prep_data("train", "mod1_train_prepped.txt", 3, tagmap, vocab)
	print prep_data("test", "mod1_test_prepped.txt", 3, tagmap, vocab)

	# to create hw3_3_1.txt
	f = open("mod1_train_prepped.txt", "r")
	lines = f.readlines()[:100]
	f.close()
	f = open("hw3_3_1.txt", "w")
	for line in lines:
		f.write(line)
	f.close()

