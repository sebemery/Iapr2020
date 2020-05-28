counter=0
while [ $counter -le 345 ]
do
	echo "Rotation: $counter degrees."	
	python3 ./main.py --input=../data/output_$counter.mp4 --output=../output/output_rotated_$counter.mp4
	counter=$[$counter +15]
done

#ffmpeg -i "concat:output_rotated_0.mp4|output_rotated_15.mp4|output_rotated_30.mp4|output_rotated_45.mp4|output_rotated_60.mp4|output_rotated_75.mp4|output_rotated_90.mp4|output_rotated_105.mp4|output_rotated_120.mp4|output_rotated_135.mp4|output_rotated_150.mp4|output_rotated_180.mp4|output_rotated_195.mp4|output_rotated_210.mp4|output_rotated_225.mp4|output_rotated_240.mp4|output_rotated_255.mp4|output_rotated_270.mp4|output_rotated_285.mp4|output_rotated_300.mp4|output_rotated_315.mp4|output_rotated_330.mp4|output_rotated_345.mp4" -c copy outputbig.mp4

#ffmpeg -f concat -i outputs.txt -codec copy bigoutput.mp4