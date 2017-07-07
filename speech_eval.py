#!/usr/bin/env python3

__author__ = "Nikolaos Malandrakis (malandra@usc.edu)"
__version__ = "$Revision: 5 $"
__date__ = "$Date: 07/06/2017$"

import os
import sys
import glob
import re
import time
import codecs
import json
import jsonschema
import Levenshtein
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import copy
import multiprocessing
import argparse
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

################################################################################
# GLOBALS
################################################################################

separators = '[,፣]' # may need to add more to accommodate new languages

mode_names = {
	'relevance' : 'Relevance',
	'type' : 'Type',
	'place' : 'Type+Place'
}

start_time = time.time()

json_schema = """
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "$version": "1.0",
  "definitions": {
      "frame": {
	  "type": "object",
	  "properties": {
	      "DocumentID": { "type": "string" },
	      "Type": { "type": "string",
			"enum": [ "Civil Unrest or Wide-spread Crime",
						"Elections and Politics",
						"Evacuation",
						"Food Supply",
						"Infrastructure",
						"Medical Assistance",
						"Shelter",
						"Terrorism or other Extreme Violence",
						"Urgent Rescue",
						"Utilities, Energy, or Sanitation",
						"Water Supply",
						"in-domain" ] },
	      "TypeConfidence": { "type": "number", "minimum": 0, "maximum": 1 },
	      "PlaceMention": { "type": "string" },
	      "Status": {
		  "type": "object",
		  "properties": {
		      "Need": {
			  "type": "string",
			  "enum": [ "Current",
						"Future",
						"Past Only" ] },
		      "Relief": {
			  "type": "string",
			  "enum": [ "Insufficient/Unknown",
						"No_Known_Resolution",
						"Sufficient" ] }
		  },
		  "required": [ "Need", "Relief" ]
	      }
	  },
	  "required": ["DocumentID", "Type", "TypeConfidence"]
      }
  },
  "type": "array",
  "items": {
       "$ref": "#/definitions/frame" 
  }
}
"""

################################################################################
# FUNCTIONS
################################################################################

# parse the Appen ground truth -------------------------------------------------
def parse_ground_truth(indir):
	print ("Parsing the Ground Truth")
	input_files = []
	rejected_files_list = []

	frames = []
	# load all input directories
	idirs = glob.glob(os.path.join(indir,'*'))
	if len(idirs) == 0:
		print ("WARNING: path "+indir+" is empty")

	for idir in idirs:
		# collect all segment annotations
		infiles = glob.glob(os.path.join(idir,'ANNOTATION','*.txt'))
		for infile in infiles:
			# parse the document id
			head, tail = os.path.split(infile)
			document_id = re.sub('\..*','',tail)
			input_files.append(document_id)
			# load the file
			f = codecs.open(infile, "r", "utf-8")
			lines = f.read().splitlines()
			f.close()


			# reject localized frames if MT+MP
			reject_file = False
			in_frames = []


			# parse
			i = 0
			while (i < len(lines)):
				if re.search('^TYPE:',lines[i]):
					# parse a frame
					fr_type = re.sub('^TYPE:\s+','',lines[i])
					fr_time = re.sub('^TIME:\s+','',lines[i+1])
					fr_resolution = re.sub('^Resolution:\s+','',lines[i+2])
					fr_place = re.sub('^PLACE:\s+','',lines[i+3])

					# check if this needs splitting
					all_types = []
					for typ in types:
						if re.search(typ,fr_type):
							all_types.append(typ)
					all_places = re.split(separators, fr_place)
					for j in range(len(all_places)):
						all_places[j] = re.sub("^\s+|\s+$",'',all_places[j])
					all_places = [x for x in all_places if x != 'n/a']

					if len(all_types) > 1 and len(all_places) > 1:
						reject_file = True

					# generate all relevant frames by cartesian product of type*place
					for typ in all_types:
						frame = {}
						frame['DocumentID'] = document_id
						frame['Type'] = typ
						if fr_time != 'n/a' or fr_resolution != 'n/a':
							frame['Status'] = {}
							if fr_time != 'n/a':
								frame['Status']['Need'] = fr_time
							if fr_resolution != 'n/a':
								frame['Status']['Relief'] = fr_resolution

						if len(all_places) == 0 or reject_file == True:
							in_frames.append(frame)
						else:
							for pl in all_places:
								frame['PlaceMention'] = pl
								in_frames.append(frame)

#					if len(all_places) > 1 and len(all_types) > 1:
#						print ('++++ '+infile)
#					if not reject_file:
					frames = frames + in_frames
					if reject_file:
						rejected_files_list.append(document_id)

					i=i+3
				i+=1

	return input_files,frames,rejected_files_list


# calculate similarity between 2 frames ----------------------------------------
def frame_similarity(frame1,frame2):
	similarity = 1
	if 'Type' in frame1:
		if frame1['Type'] != frame2['Type']:
			similarity = 0
	if similarity == 1:
		if 'PlaceMention' in frame1:
			similarity = Levenshtein.ratio(frame1['PlaceMention'], frame2['PlaceMention'])
	return similarity


# evaluate at the document level -----------------------------------------------
def document_scores(ref,out):
	# create a similarity matrix
	similarity_matrix = np.zeros((len(ref),len(out)))
	for i in range(len(ref)):
		for j in range(len(out)):
			similarity_matrix[i,j] = frame_similarity(ref[i], out[j])

	row_ind, col_ind = linear_sum_assignment(1-similarity_matrix)
	tp = similarity_matrix[row_ind, col_ind].sum()
	fp = len(out) - tp
	fn = len(ref) - tp

	return tp,fp,fn


# create a dictionary where frames are grouped by document id ------------------
def group_frames_by_document(frames):
	frames_doc = {}
	ids = []
	for frame in frames:
		doc_id = frame['DocumentID'] 
		if doc_id not in frames_doc:
			frames_doc[doc_id] = []
			ids.append(doc_id)
		frames_doc[doc_id].append(frame)
	return frames_doc, ids


# evaluate at the frameset level -----------------------------------------------
def frameset_scores(ref,out):
	ref_doc, ref_ids = group_frames_by_document(ref)
	out_doc, out_ids = group_frames_by_document(out)
	all_ids = sorted(list(set(ref_ids + out_ids)))

	tp = 0
	fn = 0
	fp = 0
	for doc_id in all_ids:
#		print (doc_id + "=================================================")
		if doc_id in ref_ids:
			if doc_id in out_ids:
				tp1,fp1,fn1 = document_scores(ref_doc[doc_id],out_doc[doc_id])
				tp += tp1
				fp += fp1
				fn += fn1
			else:
				fn = fn + len(ref_doc[doc_id])
		else:
			fp = fp + len(out_doc[doc_id])
#		print([tp,fp,fn])
	prec  = 0
	rec   = 0
	if (tp + fp) > 0:
		prec = float(tp) / (tp + fp)
	if (tp + fn) > 0:
		rec = float(tp) / (tp + fn)
	if rec == 0:
		prec = 1
	return tp,fp,fn,prec,rec


# reform the frames to the different equivalence classes -----------------------
def equivalence_classes(frames,mode):
	out_frames = []
	if mode == 'all':
		out_frames = copy.deepcopy(frames)
	else:
		for frame in frames:
			new_frame = None
			if mode == 'relevance':
				new_frame = {'DocumentID': frame['DocumentID'], 'Type':'in-domain'}
			elif mode == 'type':
				new_frame = {'DocumentID': frame['DocumentID'], 'Type':frame['Type']}
			elif mode == 'place':
				if 'PlaceMention' in frame:
					new_frame = {'DocumentID': frame['DocumentID'], 'Type':frame['Type'], 'PlaceMention':frame['PlaceMention']}
			if new_frame is not None:
				if new_frame not in out_frames:
					out_frames.append(new_frame)
	return out_frames


# parse json schema ------------------------------------------------------------
def parse_schema():
	# load json schema
	schema = json.loads(json_schema)

	# label vocabularies
	types = schema['definitions']['frame']['properties']['Type']['enum']
	times = schema['definitions']['frame']['properties']['Status']['properties']['Need']['enum']
	resolutions = schema['definitions']['frame']['properties']['Status']['properties']['Relief']['enum']

	return schema, types, times, resolutions


# a point of the PR curve | to be used with multi-processing -------------------
def single_point(params):
	reference_frames = params[0]
	output_frames = params[1]
	threshold = params[2]
	mode = params[3]

	out_frames = output_frames[:threshold]
	# convert gt and output to equivalence class
	ref_frames = equivalence_classes(reference_frames,mode)
	out_frames = equivalence_classes(out_frames,mode)

	if threshold < 0:
		print("TOTAL FRAMES @0 = %d" % len(out_frames))

	if threshold == 0.5:
		print("TOTAL FRAMES @0.5 = %d" % len(out_frames))

	tp,fp,fn,prec,rec = frameset_scores(ref_frames,out_frames)

	return tp,fp,fn,prec,rec


# create PR curve --------------------------------------------------------------
def get_pr(reference_frames,output_frames,mode='type',pr_resolution=100):

	# setup an exponential sampling window
	step_factor = math.pow(len(output_frames),1/float(pr_resolution))
	conf_order = []
	for j in range(1,pr_resolution+1):
		tmp = round(math.pow(step_factor,j))
		if tmp < len(output_frames):
			if tmp not in conf_order:
				conf_order.append(tmp)
	conf_order = [0] + conf_order + [len(output_frames)]
	print(len(conf_order))

	# get curve
	params = []
	for threshold in conf_order:
		params.append( [ reference_frames, output_frames, threshold, mode ] )
	all_tp, all_fp, all_fn, all_prec, all_rec = zip(*pool.map(single_point, params))
	all_prec = list(all_prec) #+ [0]
	all_rec  = list(all_rec) #+ [1]

	all_rec, all_prec = zip(*sorted(zip(all_rec, all_prec)))

	# remove 0 recall points
	all_rec1 = []
	all_prec1= []
	for i in range(len(all_rec)):
		if all_rec[i] > 0:
			all_rec1.append(all_rec[i])
			all_prec1.append(all_prec[i])
	all_rec = all_rec1
	all_prec= all_prec1

	if len(all_rec) > 0:
		AUC = metrics.auc(all_rec, all_prec)
	else:
		AUC = 0

	return all_rec, all_prec, AUC


# create complete output for 1 mode --------------------------------------------
def get_complete_output(reference_frames,output_frames,mode,pr_resolution,outdir):

	print ("Processing Layer: %s" % mode_names[mode])
	start_time = time.time()

	# sort system frames by confidence score, descending
	output_frames = sorted(output_frames, key=lambda k: k['TypeConfidence'],reverse=True) 

	# get the curve
	pr_x, pr_y, pr_AUC= get_pr(reference_frames,output_frames,mode,pr_resolution)

	# get maximum f-score
	f_score = []
	for i in range(len(pr_x)):
		f_score.append( (2*pr_x[i]*pr_y[i])/(pr_x[i]+pr_y[i]) )
	if len(f_score) > 0:
		f = max(f_score)
	else:
		f = 0

	# create a plot
	plt.plot(pr_x,pr_y)
	plt.title(mode_names[mode])
	plt.xlabel('recall')
	plt.ylabel('precision')
	plt.grid()
	ax = plt.gca()
	ax.set_ylim([-0.05, 1.05])
	ax.set_xlim([-0.05, 1.05])
	ax.set(adjustable='box-forced', aspect='equal')
	gc = plt.gcf()
	gc.set_size_inches(7, 7)
	str1 = "AUC=%.3f, max F1=%.3f" % (pr_AUC,f)
	plt.legend([str1], loc='upper right')
	pp = PdfPages(os.path.join(outdir,'curve_'+mode_names[mode]+'.pdf'))
	pp.savefig(plt.gcf())
	pp.close()
	plt.close()

	# save complete log
	arr = np.array([pr_x,pr_y])
	np.savetxt(os.path.join(outdir,'log_'+mode_names[mode]+'.tsv'), np.transpose(arr), fmt='%.8f', delimiter="\t", header="recall\tprecision", comments='')

	print("MAX F = %.3f" % f)
	print("AUC = %.3f" % pr_AUC)
	print("Done --- %s seconds ---" % (time.time() - start_time))

	return pr_x, pr_y, pr_AUC




################################################################################
# MAIN
################################################################################

if __name__ == "__main__":

	debug = False

	# argument parser ----------------------------------------------------------
	if not debug:
		parser = argparse.ArgumentParser(description='Generates PR curves for the LORELEI speech evaluation')
		parser.add_argument('-s', '--system-output', help='Path to the system output', required=True)
		parser.add_argument('-g', '--ground-truth', help='Path to the ground truth directory', required=True)
		parser.add_argument('-o', '--output-directory', help='Path to save the evaluation output', required=True)

		args = parser.parse_args()
#		print (args)

		ref_dir  = args.ground_truth
		test_file = args.system_output
		outdir = args.output_directory
		pr_resolution = 100 # points used to approximate the PR curve
	else:
		ref_dir  = 'ground_truth'
		test_file = 'system_output.json'
		outdir = 'output'
		pr_resolution = 10 # points used to approximate the PR curve

	# create output directory
	try:
		if not os.path.exists(outdir): os.makedirs(outdir) 
	except:
		sys.exit('CAN NOT CREATE OUTPUT DIRECTORY: '+outdir)

	# multi-processing pool
	pool = multiprocessing.Pool(8)


	# basic error checks -------------------------------------------------------
	if not os.path.exists(ref_dir):
		sys.exit('PATH NOT FOUND: '+ref_dir)

	if not os.path.exists(test_file):
		sys.exit('PATH NOT FOUND: '+test_file)

	# parse the input data -----------------------------------------------------
	# parse the schema
	schema, types, times, resolutions = parse_schema()
	# load the ground truth
	input_files,reference_frames,rejected_files_list = parse_ground_truth(ref_dir)
	# load test file
	fp = codecs.open(test_file, "r", "utf-8")
	output_frames = json.load(fp)
	fp.close()

	# validate system output
	try:
		jsonschema.validate(output_frames, schema)
	except jsonschema.ValidationError as e:
		print (e.message)
	except jsonschema.SchemaError as e:
		print (e)

	# remove location entries from any frame of a rejected document
	for i in range(len(output_frames)):
		if output_frames[i]['DocumentID'] in rejected_files_list:
			if 'PlaceMention' in output_frames[i]:
				del output_frames[i]['PlaceMention']

	# check which layers were submitted ----------------------------------------
	modes = ['relevance']

	# check for types
	has_type = False
	for frame in output_frames:
		if frame['Type'] != 'in-domain':
			has_type = True
	if has_type:
		modes.append('type')

	# check for any placementions
	has_place = False
	if has_type:
		for frame in output_frames:
			if 'PlaceMention' in frame:
				has_place = True
		if has_place:
			modes.append('place')

	# get the complete output --------------------------------------------------
	all_AUC = {}
	for mode in modes:
		pr_x, pr_y, pr_AUC = get_complete_output(reference_frames,output_frames,mode,pr_resolution,outdir)
		all_AUC[mode_names[mode]] = pr_AUC

	# print Summary
	f = open(os.path.join(outdir,'summary.tsv'),'w')
	f.write("%s\t%s\n" % ('Layer','AUC'))
	for mode in modes:
		f.write("%s\t%.8f\n" % (mode_names[mode],all_AUC[mode_names[mode]]))
	f.close()
