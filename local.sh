data="example"
minirun=0 #0=false
a=1
b=1
c=1
T=1
predictive=0 #0=false any other number is true

echo 'Starting Fibril tracking Script a1'
python a1_fibtrackMain.py $data $minirun $a $b $c $T $predictive
echo 'Completed Fibril tracking script a1'

python a2_fibtrackStats.py $data $minirun $a $b $c $T $predictive
# echo 'complete 2/3'
