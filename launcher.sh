rm ./notebooks/out.txt

for i in "1" "3" "4" "5"; do
    python3 main.py -m pr -t 2 -c $1 -q $i
    sleep 5
done

for i in `seq 1 5`; do
    python3 main.py -m pr -t 1 -c $1 -q $i
    sleep 5
done

# for i in `seq 1 99`; do
#     python3 main.py -m te -i $i
# done
