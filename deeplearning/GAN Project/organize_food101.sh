!wget "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
!tar -xvzf food-101.tar.gz

mkdir food-101/images/train
mkdir food-101/images/test
while read p; do
  mkdir food-101/images/train/$p
  mkdir food-101/images/test/$p
done <classes.txt

while read p; do
	mv food-101/images/$p.jpg food-101/images/test/$(basename $(dirname $p))
done <food-101/meta/test.txt

while read p; do
	mv food-101/images/$p.jpg food-101/images/train/$(basename $(dirname $p))
done <food-101/meta/train.txt