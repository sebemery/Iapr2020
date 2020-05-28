counter=0
while [ $counter -le 345 ]
do
	echo "Rotation: $counter degrees."	
	ffmpeg -i ./robot_parcours_1.avi -c:v libx265 -crf 2 -vf "rotate=$counter*PI/180:ow='max(iw,ih)':oh='max(iw,ih)'" ./output_$counter.mp4
	counter=$[$counter +15]
done

