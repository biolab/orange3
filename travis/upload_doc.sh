cd $TRAVIS_BUILD_DIR

#echo -e "Host butler.fri.uni-lj.si\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config
#scp -i travis/key.private -P 5722 -r doc/build/html/ travis@butler.fri.uni-lj.si:/home/travis/html
ln -s `pwd`/doc/build/html doc/build/orange3doc
echo -e "Host orange.biolab.si\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config
echo -e "\tUser uploaddocs\n\tIdentityFile $TRAVIS_BUILD_DIR/travis/key.private\n" >> ~/.ssh/config
rsync -a --delete doc/build/orange3doc/ orange.biolab.si:/orange3doc/
