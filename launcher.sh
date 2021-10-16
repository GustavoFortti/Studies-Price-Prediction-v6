# rm ./data/treined/$1/data_predict/test.csv

for j in `seq 5 7`; do
    for i in `seq 0 3`; do
        python3 main.py -m te -t 2 -c XOM -q $j -i 10 -tah $i
        sleep 3
    done
done
