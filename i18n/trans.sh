if [ "$#" -ne 2 ]
then
    echo "trans <language> <destination>"
    exit
else
    lang=$1
    dest=$2
    trubar --conf $lang/trubar-config.yaml translate -s ../Orange -d $dest/Orange --static $lang/static $lang/msgs.jaml
    trubar --conf $lang/tests-config.yaml translate -s ../Orange -d $dest/Orange $lang/tests-msgs.jaml
fi
