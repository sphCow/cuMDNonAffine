for i in `find . -maxdepth 2`
do
echo "---------------------------"
echo $i
cat $i | grep "128"
echo "---------------------------"
done
