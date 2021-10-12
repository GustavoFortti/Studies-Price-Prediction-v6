# rm ./data/treined/$1/data_predict/test.csv

for i in `seq 1 30`; do
    python3 main.py -m te -t 2 -c XOM -q 1 -i $(( $i + 1 )) -tah 1
    sleep 2
    python3 main.py -m te -t 2 -c XOM -q 1 -i $(( $i + 2 )) -tah 2
    sleep 2
    python3 main.py -m te -t 2 -c XOM -q 1 -i $(( $i + 3 )) -tah 3
    sleep 2
    python3 main.py -m te -t 2 -c XOM -q 1 -i $(( $i + 4 )) -tah 4
    sleep 2
    python3 main.py -m te -t 2 -c XOM -q 1 -i $(( $i + 5 )) -tah 5
    sleep 2
done

# cp ./data/treined/$1/data_predict/test.csv ./notebooks

# for i in `seq 1 4`; do
#     python3 main.py -m tr -t 2 -c $1 -q $i -i 1
#     sleep 5
# done

