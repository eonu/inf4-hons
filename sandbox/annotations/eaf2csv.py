import csv
import re
import sys

# .eaf text to data
def r_eaf(inputPath):
	# re template
	timeId_re = re.compile('TIME_SLOT_ID="(.+)" T')
	timeValue_re = re.compile('TIME_VALUE="(.+)"')
	annotationValue_re = re.compile('>(.+)<')
	timeRef1_re = re.compile('TIME_SLOT_REF1="(.+)" T')
	timeRef2_re = re.compile('TIME_SLOT_REF2="(.+)"')

	ref = {}
	data = []
	prev_line = ""
	with open(inputPath,'rb') as f:
		for line in f:
			# get the reference of time slots
			if 'TIME_SLOT_ID' in line:
				line = line.strip()
				timeId = re.findall(timeId_re,line)
				timeValue = re.findall(timeValue_re,line)
				timeId = timeId[0]
				timeValue = timeValue[0]
				ref[timeId] = timeValue

			# match the time slots with annotations
			elif 'ANNOTATION_VALUE' in line:
				line = line.strip()
				label = re.findall(annotationValue_re,line)
				# label and speaking
				try:
					label = label[0]
				except:
					import pdb
					pdb.set_trace()
				speaking = 1
				if '0' in label:
					speaking = 0
					label = label[3:]
				# get time information from previous line traced 
				timeRef1 = re.findall(timeRef1_re,prev_line)
				timeRef2 = re.findall(timeRef2_re,prev_line)
				timeRef1 = timeRef1[0]
				timeRef2 = timeRef2[0]
				# time
				start = ref[timeRef1]
				end = ref[timeRef2]
				during = int(end) - int(start)
				data.append([label,speaking,start,end,during])

			# track previous line
			prev_line = line

	# print data

	return data



# write csv file
def w_csv(savePath, data):
	with open(savePath, 'w') as csvfile:
	    fieldnames = ['type', 'speaking', 'start_time', 'end_time', 'during_time']
	    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

	    writer.writeheader()
	    for subData in data:
	    	writer.writerow({'type': str(subData[0]), 'speaking': str(subData[1]), 'start_time': str(subData[2]), 'end_time': str(subData[3]), 'during_time': str(subData[4])})


if __name__ == '__main__':
	inputPath = sys.argv[1]
	outputPath = sys.argv[2]
	data = r_eaf(inputPath = inputPath) 
	w_csv(savePath = outputPath, data = data)
	print "input path:", inputPath
	print "output path:", outputPath
	print "eaf to csv success......."




