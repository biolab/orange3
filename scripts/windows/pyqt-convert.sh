#!/bin/bash -e


function print_usage {
    echo 'pyqt4-convert.sh INSTALLER
Convert a PyQt4 windows installer into a wheel package.

note: 7z executable must be on PATH.

Options:
    -d -dist-dir DIR    Directory where the build .whl package is put (the
                        default is "dist")
    -h --help           Print this help and exit.
'
}

while [ ${1:0:1} == "-" ]
do
    case $1 in
        -d|--dist-dir)
            DISTDIR=$2
            shift 2;;
        -h|-help)
            print_usage
            exit 0 ;;
        -*)
            echo "Unrecognized option $1"
            print_usage
            exit 1 ;;
    esac
done

INSTALLER=$1
DISTDIR=${DISTDIR:-dist}

if [[ ! -f "$INSTALLER" ]]
then
    echo "$INSTALLER does not exist"
    print_usage
    exit 1
fi

INSTBASE=$(basename "$INSTALLER")

# Get the PyQt4 version, python version, and platform tag from the installer
# filename.
# PyQt4-{VERSION}-Py{PYVER}-{QTVER}-(x32|x64).exe
VERSION=$(echo "$INSTBASE" | cut -d "-" -f 2)
PYVERSION=$(echo "$INSTBASE" | cut -d "-" -f 4 | sed s/\\.//g | awk '{print tolower($0)}')
PYTAG=cp${PYVERSION##py}
PLATTAG=$(echo "$INSTBASE" | cut -d "-" -f 6 | cut -d "." -f 1)

case $PLATTAG in
    x32)
        PLATTAG=win32
        ;;
    x64)
        PLATTAG=win_amd64
        ;;
    *)
        echo "Unrecognized platform tag $PLATTAG"
        exit 1;
esac


mkdir -p "$DISTDIR"

function extract-installer {
    local installer=${1:?}
    local wheelbase=${2:?}

    local tmpdir=$(mktemp -d -t pyqt4)

    7z -o"${tmpdir:?}" -y x "$installer"

    cp -a -f "$tmpdir"/Lib/site-packages/* "$wheelbase"/

	if [[ -d "$tmpdir"/'$_OUTDIR' ]]; then
		# * .pyd, doc/, examples/, mkspecs/, qsci/, include/, uic/
		cp -a -f "$tmpdir"/'$_OUTDIR'/*.pyd "$wheelbase"/PyQt4/
        cp -a -f "$tmpdir"/'$_OUTDIR'/uic "$wheelbase"/PyQt4/
        # ignore the rest
	fi
    rm -r "$tmpdir"
}


function wheel-convert {
    local installer=${1:?}
    local distdir=${2:?}

    local version=$VERSION
    local pytag=$PYTAG
    local plattag=$PLATTAG

    mkdir -p "$distdir"

    local wheelbase=$(mktemp -d -t pyqt4-wheel-convert)

    mkdir -p "${wheelbase:?}"/PyQt4
    mkdir -p "$wheelbase"/PyQt4-${version}.dist-info
    mkdir -p "$wheelbase"/PyQt4-${version}.data/data

    extract-installer "$installer" "$wheelbase"

    echo '[PATHS]
Prefix = Lib\\site-packages\\PyQt4
' > "$wheelbase"/PyQt4-${version}.data/data/qt.conf

    echo '[PATHS]
Prefix = .
' > "$wheelbase"/PyQt4/qt.conf

    echo "Wheel-Version: 1.0
Generator: pyqt-convert.sh
Root-Is-Purelib: false
Tag: ${pytag}-none-${plattag}
Build: 1" > "${wheelbase}"/PyQt4-${version}.dist-info/WHEEL

    echo "Metadata-Version: 1.1
Name: PyQt4
Version: ${version}
Summary: xxx
Home-page: xxx
Author: xxx
Author-email: xxx
License: GPLv3
Download-URL: xxx
Description: xxx
" > "${wheelbase}"/PyQt4-${version}.dist-info/METADATA

    generate_record "$wheelbase" > "$wheelbase"/PyQt4-${version}.dist-info/RECORD

    echo "PyQt4-${version}.dist-info/RECORD,," >> "$wheelbase"/PyQt4-${version}.dist-info/RECORD
    local wheelname=PyQt4-${version}-${pytag}-none-${plattag}.whl

    wheel-zip "$wheelbase" "$wheelname"
    mv "$wheelbase/$wheelname" "$distdir"
}


function wheel-zip {
    local wheelbase=${1:?}
    local wheelname=${2:?}
    wheelbase=$(cd "${wheelbase:?}"; pwd;)
    (cd "$wheelbase"; find . -print | zip "$wheelname" -@)
}

function find_all {
    (cd "${1:?}"; find . ;)
}


function urlsafe_b64encode_nopad {
    python3 -c "import base64; print(base64.urlsafe_b64encode(bytes.fromhex('$1')).rstrip(b'=').decode())"
}


function generate_record {
    local wheelbase=$1
    find_all "$wheelbase" | while read line
    do
        line=${line##./}  # strip the leading ./..
        local filepath="$wheelbase"/"$line"
        if [[ -f "$filepath" ]]
        then
            local sha=$(_sha256sum "$filepath")
            local size=$(stat -f "%z" "$filepath")
            echo "$line,sha256=$sha,$size"
        fi
    done
}

# _sha256sum PATH
# Return the sha256 checksum of PATH encoded with urlsafe_b64encode
# suitable for wheel RECORD file (see PEP 427 for details)
function _sha256sum {
    python -c"
import sys, hashlib, base64
sha = hashlib.sha256()
for line in open(sys.argv[1], \"rb\"):
    sha.update(line)
print(base64.urlsafe_b64encode(sha.digest()).rstrip(b'=').decode())
" "${1:?}"
}

wheel-convert "$INSTALLER" "$DISTDIR"
