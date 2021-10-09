# rm ./data/treined/$1/data_predict/test.csv

for i in `seq 6 28`; do
    python3 main.py -m te -t 2 -c XOM -q 3 -i $i
    sleep 5
done

# cp ./data/treined/$1/data_predict/test.csv ./notebooks

# for i in `seq 1 4`; do
#     python3 main.py -m tr -t 2 -c $1 -q $i -i 1
#     sleep 5
# done