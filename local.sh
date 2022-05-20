data="9am-1R"
minirun=0 # 0=false any other number is true
a=0.5
b=1
c=1.5
T=10

# python3 a1_fibtrackMain.py $data $minirun $a $b $c $T
# echo 'complete 1/3'

# python3 a2_fibtrackStats.py $data $minirun $a $b $c $T
# echo 'complete 2/3'
#
python3 b2_volumeRendering.py $data $minirun $a $b $c $T
echo 'complete 3/3'
say howdy
