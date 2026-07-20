## code
interaction=${1}

# Iterate over each value of electric field and number density:
for v in $(seq 25 5 200); do
	for n in $(seq 1e11 .4e11 1.5e12); do
        	"uv" "run" "main.py" $v $n $interaction
	done
done
